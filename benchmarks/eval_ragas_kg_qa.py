r"""Ragas evaluation of QAAgent's real answer-composition pipeline.

Why this script exists (and how it differs from eval_kg_qa.py)
-----------------------------------------------------------------
`benchmarks/eval_kg_qa.py` generates questions from the graph and answers them
with the graph-traversal primitive directly (`LocalGraphStore.neighbors`), never
touching `QAAgent.answer()`'s LLM composition step -- so its 1.000/1.000/1.000
is a self-consistency check on retrieval, not a measurement of answer quality.
`docs/EVIDENCE_MAP.md` now labels it accordingly.

This script is the independent measurement: it loads the human/grounded
question set from `benchmarks/build_kg_qa_human.py` (references extracted from
and verified against `graph_scaled.json` at generation time, not hand-typed),
runs each question through the *real* `QAAgent.answer()` (NER anchor resolution
-> subgraph retrieval -> LLM-composed prose), and scores the actual returned
answer with Ragas:

  * faithfulness         -- is the answer's content supported by the retrieved
                             evidence, or did the model add unsupported claims?
  * answer_relevancy     -- does the answer actually address the question asked?
  * context_precision    -- how much of the retrieved evidence was relevant?
  * context_recall       -- did retrieval surface what the reference answer needed?

Judge model: whatever `LLMClient` is configured with (`.env`'s `LLM_PROVIDER`),
reused for both the Ragas judge and the embeddings model -- no separate judge
config to keep in sync. Ragas is LLM-as-judge, which has known reliability
limits; this is reported as a *complement* to, not a replacement for,
deterministic checks (the safe-rejection rate below is computed by simple
string/boolean logic, not judged).

    python benchmarks/eval_ragas_kg_qa.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import types
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ragas 0.4.3's llms/base.py unconditionally imports langchain_community's
# VertexAI integration; the installed langchain-community (post-split into
# langchain-google-vertexai) no longer ships it. We never use VertexAI -- stub
# it so the rest of ragas (which we do use) imports cleanly. See
# docs/EVALUATION_PROTOCOL.md's Ragas section for the full story.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub
import langchain_community.llms as _llms_mod  # noqa: E402

if not hasattr(_llms_mod, "VertexAI"):
    _llms_mod.VertexAI = type("VertexAI", (), {})

from ragas.embeddings import OpenAIEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.agents.qa_agent import QAAgent, _relation_label  # noqa: E402
from medigraph.graph.local_store import LocalGraphStore  # noqa: E402
from medigraph.llm.client import LLMClient  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

DATASET = Path(__file__).resolve().parent / "kg_qa_human.json"
GRAPH = Path(OUTPUTS_DIR) / "graph_scaled.json"
OUT = Path(OUTPUTS_DIR) / "eval_ragas_kg_qa.json"


ANSWER_TIMEOUT_S = 180
#: `score_one` makes 4 sequential judge calls; with a smaller/cheaper judge
#: model given a generous max_tokens budget (see `llm_factory(..., max_tokens=)`
#: below) each call can legitimately take 1-2 minutes of real generation time,
#: so the 4-call chain needs a much longer ceiling than a single answer call.
SCORING_TIMEOUT_S = 600
#: `LLMClient`'s own retry loop already bounds each individual HTTP call by
#: `LLM_TIMEOUT`, but a rare stalled connection (observed in practice: zero
#: CPU progress for 30+ minutes on a single question) can sit below that --
#: e.g. a TCP connection that never reads and never errors. Wrapping each
#: `agent.answer()` call in a one-shot thread with a wall-clock deadline
#: (same pattern as `medigraph/agents/dag_executor.py`'s `_call`) guarantees
#: one bad question can cost at most `ANSWER_TIMEOUT_S`, not the whole run.
#: The abandoned thread is left to finish or die on its own; the process
#: exits normally at the end regardless (daemon-like: no join is required).
def answer_with_timeout(agent: QAAgent, question: str, timeout: float = ANSWER_TIMEOUT_S) -> dict:
    # One-off worker, never joined: a `with ThreadPoolExecutor(...)` block would
    # call shutdown(wait=True) on exit and block anyway on the very timeout
    # we're trying to escape. shutdown(wait=False) abandons the stuck thread
    # instead (mirrors medigraph/agents/dag_executor.py's `_call`).
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas-eval-timeout")
    future = pool.submit(agent.answer, question, verbose=False)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        return {"answer": "", "evidence": [], "refused": False, "timed_out": True}
    finally:
        pool.shutdown(wait=False)


def evidence_to_contexts(evidence: list[dict]) -> list[str]:
    return [
        f"{item['head']} --[{_relation_label(item)}]--> {item['tail']}"
        for item in evidence
    ]


def entity_mention_recall(answer: str, gold_entities: list[str]) -> float | None:
    """Cheap, non-LLM cross-check: fraction of gold entities literally quoted in
    the answer text. Not a substitute for Ragas's judged context_recall -- a
    correct paraphrase would score 0 here -- but it costs nothing and catches
    the case where the answer clearly never engaged with the evidence at all."""
    if not gold_entities:
        return None
    hits = sum(1 for entity in gold_entities if entity in answer)
    return round(hits / len(gold_entities), 4)


async def _score_metric(name: str, coro) -> tuple[str, float | None, str | None]:
    try:
        result = await coro
        return name, float(result.value), None
    except Exception as exc:  # noqa: BLE001 - one bad judge call must not kill the run
        return name, None, str(exc)[:300]


async def score_one(metrics: dict, sample: dict, answer: str, contexts: list[str]) -> dict:
    # The 4 metrics are independent judge calls (no shared state) -- run them
    # concurrently rather than one after another. Sequential execution was the
    # actual bottleneck (each call can take 1-2 minutes with a generous
    # max_tokens budget): 4 calls back-to-back vs. in parallel is roughly a 4x
    # wall-clock difference for what's an I/O-bound wait, not CPU work.
    results = await asyncio.gather(
        _score_metric(
            "faithfulness",
            metrics["faithfulness"].ascore(user_input=sample["question"], response=answer, retrieved_contexts=contexts),
        ),
        _score_metric(
            "answer_relevancy",
            metrics["answer_relevancy"].ascore(user_input=sample["question"], response=answer),
        ),
        _score_metric(
            "context_precision",
            metrics["context_precision"].ascore(
                user_input=sample["question"], reference=sample["reference"], retrieved_contexts=contexts
            ),
        ),
        _score_metric(
            "context_recall",
            metrics["context_recall"].ascore(
                user_input=sample["question"], retrieved_contexts=contexts, reference=sample["reference"]
            ),
        ),
    )
    scores: dict = {}
    for name, value, error in results:
        if error is None:
            scores[name] = value
        else:
            scores[f"{name}_error"] = error
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-per-hop",
        type=int,
        default=None,
        help="cap positives to the first N per hop depth (stratified subsample instead of the full grounded set)",
    )
    parser.add_argument(
        "--max-negative",
        type=int,
        default=None,
        help="cap the number of out-of-graph safe-rejection questions scored",
    )
    return parser.parse_args()


async def main_async() -> None:
    enable_utf8()
    args = parse_args()
    if not DATASET.exists():
        raise SystemExit(f"{DATASET} missing; run benchmarks/build_kg_qa_human.py first")
    if not GRAPH.exists():
        raise SystemExit(f"{GRAPH} missing; build it first (scripts/build_scaled_kg.py)")

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    samples = dataset["samples"]
    positives = [s for s in samples if s["category"] != "safe_rejection"]
    negatives = [s for s in samples if s["category"] == "safe_rejection"]
    all_positives, all_negatives = positives, negatives

    if args.limit_per_hop is not None:
        kept: list[dict] = []
        seen_per_hop: dict[int, int] = {}
        for sample in positives:
            hop = sample["hops"]
            if seen_per_hop.get(hop, 0) >= args.limit_per_hop:
                continue
            seen_per_hop[hop] = seen_per_hop.get(hop, 0) + 1
            kept.append(sample)
        positives = kept
    if args.max_negative is not None:
        negatives = negatives[: args.max_negative]

    llm = LLMClient()
    agent = QAAgent(llm=llm, store=LocalGraphStore.load_json(GRAPH), hops=2)

    # A separate AsyncOpenAI client for the judge/embeddings, distinct from
    # `llm.client` (sync `openai.OpenAI`, used by QAAgent.answer() via
    # LLMClient.chat()). ragas's collections API is async-only (`.ascore()`);
    # passing a sync client to llm_factory and then calling `.ascore()` fails
    # with "Cannot use agenerate() with a synchronous client" -- silently, from
    # this script's perspective, since each metric call is individually try/
    # excepted so the run completes with every score simply absent rather than
    # crashing loudly. Same api_key/base_url/timeout as the sync client, so
    # this is the same provider account, just the async transport.
    from openai import AsyncOpenAI

    async_client = AsyncOpenAI(
        api_key=llm.config.api_key or "EMPTY",
        base_url=llm.config.base_url,
        timeout=llm.config.timeout,
    )
    # Ragas's structured-output prompts (claim decomposition, per-context verdicts)
    # can run long; without an explicit max_tokens the judge call inherits
    # whatever short default the provider applies, and smaller/cheaper judge
    # models silently truncate before emitting valid JSON -- ragas then reports
    # "output is incomplete due to a max_tokens length limit" and the metric
    # comes back as None for nearly every sample.
    judge = llm_factory(model=llm.config.model, client=async_client, max_tokens=8192)
    embeddings = OpenAIEmbeddings(client=async_client, model=llm.config.embedding_model)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }

    detail: list[dict] = []
    print(f"[1/2] answering {len(positives)} positive + {len(negatives)} negative questions ...", flush=True)
    for index, sample in enumerate(positives, start=1):
        result = answer_with_timeout(agent, sample["question"])
        contexts = evidence_to_contexts(result.get("evidence", []))
        row = {
            "question": sample["question"],
            "category": sample["category"],
            "hops": sample["hops"],
            "refused": result.get("refused", False),
            "answer": result.get("answer", ""),
            "retrieved_contexts": contexts,
            "reference": sample["reference"],
            "entity_mention_recall": entity_mention_recall(result.get("answer", ""), sample["gold_entities"]),
        }
        if result.get("timed_out"):
            row["ragas_skipped_reason"] = f"agent.answer() exceeded {ANSWER_TIMEOUT_S}s wall-clock"
        elif result.get("refused") or not contexts:
            row["ragas_skipped_reason"] = "agent refused or returned no evidence"
        else:
            try:
                row["ragas"] = await asyncio.wait_for(
                    score_one(metrics, sample, row["answer"], contexts), timeout=SCORING_TIMEOUT_S
                )
            except TimeoutError:
                row["ragas_skipped_reason"] = f"ragas judge scoring exceeded {SCORING_TIMEOUT_S}s wall-clock"
        detail.append(row)
        status = "TIMEOUT" if result.get("timed_out") else ("REFUSED" if row["refused"] else "ok")
        print(f"  [{index}/{len(positives)}] {status:8} {sample['question'][:36]}", flush=True)

    print(f"[2/2] checking {len(negatives)} out-of-graph negatives (safe rejection) ...", flush=True)
    negative_detail: list[dict] = []
    for sample in negatives:
        result = answer_with_timeout(agent, sample["question"])
        negative_detail.append(
            {
                "question": sample["question"],
                "refused": result.get("refused", False),
                "answer": result.get("answer", ""),
                "timed_out": result.get("timed_out", False),
            }
        )

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 4) if values else None

    ragas_rows = [row["ragas"] for row in detail if "ragas" in row]
    aggregate = {
        metric: mean([row[metric] for row in ragas_rows if metric in row])
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    }
    entity_recalls = [row["entity_mention_recall"] for row in detail if row.get("entity_mention_recall") is not None]

    report = {
        "judge_model": llm.config.model,
        "judge_provider": llm.config.provider,
        "graph_file": str(GRAPH),
        "dataset_file": str(DATASET),
        "sample_mode": (
            f"stratified subsample: limit_per_hop={args.limit_per_hop}, max_negative={args.max_negative} "
            f"(full grounded set is {len(all_positives)} positive + {len(all_negatives)} negative)"
            if args.limit_per_hop is not None or args.max_negative is not None
            else "full grounded set"
        ),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "n_scored": len(ragas_rows),
        "n_refused_or_no_evidence": sum(1 for row in detail if "ragas" not in row),
        "ragas_aggregate": aggregate,
        "entity_mention_recall_mean": mean(entity_recalls),
        "safe_rejection_rate": round(
            sum(1 for row in negative_detail if row["refused"]) / len(negative_detail), 4
        )
        if negative_detail
        else None,
        "by_hops": {
            str(hops): {
                "n": len([row for row in detail if row["hops"] == hops]),
                "faithfulness": mean(
                    [row["ragas"]["faithfulness"] for row in detail if row["hops"] == hops and "ragas" in row and "faithfulness" in row["ragas"]]
                ),
                "context_recall": mean(
                    [row["ragas"]["context_recall"] for row in detail if row["hops"] == hops and "ragas" in row and "context_recall" in row["ragas"]]
                ),
            }
            for hops in sorted({row["hops"] for row in detail})
        },
        "note": (
            "Ragas is LLM-as-judge (same provider as the answering agent here); "
            "report alongside, not instead of, deterministic checks. "
            "entity_mention_recall is a cheap non-LLM cross-check (literal string "
            "containment), not a substitute for context_recall -- a correct "
            "paraphrase scores 0 on it."
        ),
        "detail": detail,
        "negative_detail": negative_detail,
    }
    path = write_json(report, OUT)

    print("\n================ RAGAS (GraphRAG QA) ================")
    for metric, value in aggregate.items():
        print(f"  {metric:20} {value}")
    print(f"  entity_mention_recall (non-LLM) {report['entity_mention_recall_mean']}")
    print(f"  safe_rejection_rate             {report['safe_rejection_rate']}  ({len(negatives)} out-of-graph questions)")
    print(f"  scored {len(ragas_rows)}/{len(positives)} positives ({report['n_refused_or_no_evidence']} refused/no-evidence)")
    print(f"  (saved -> {path})")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
