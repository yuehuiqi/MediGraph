"""Request/response models for the HTTP API.

Kept separate from the medigraph package so the library stays usable offline
without FastAPI/Pydantic-v2 web types leaking into it.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000, description="医疗文本")
    backend: Literal["auto", "neural", "fast", "llm"] | None = Field(
        None, description="覆盖 EXTRACTION_BACKEND；缺省用服务端配置"
    )
    link_entities: bool = Field(True, description="是否做实体链接与规范 ID")


class Entity(BaseModel):
    """Mirrors the `medical_ner` / `entity_linker` operator output.

    `extra="allow"` keeps operator-specific fields (offsets, match_method, score)
    from being silently dropped when the operators evolve, while the declared
    fields still document the contract in the OpenAPI schema.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    type: str
    confidence: float | None = None
    canonical_id: str | None = None
    canonical_name: str | None = None


class Triple(BaseModel):
    model_config = ConfigDict(extra="allow")

    head: str
    relation: str
    tail: str
    confidence: float | None = None
    predicate: str | None = None


class ExtractResponse(BaseModel):
    entities: list[Entity]
    triples: list[Triple]
    routing: dict[str, Any] = Field(
        default_factory=dict,
        description="级联路由证据：命中层级（L1_neural / L1_fast / L2_llm）、是否调用 LLM、模型版本",
    )
    linking_report: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# knowledge graph
# --------------------------------------------------------------------------- #
class KGBuildRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    graph_name: str = Field("api_graph.json", description="输出图谱文件名（限 outputs/ 下）")


class KGBuildResponse(BaseModel):
    graph_name: str
    num_entities: int
    num_triples: int
    validation: dict[str, Any]
    elapsed_ms: float


class KGQARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1_000)
    graph_name: str = Field("graph.json")
    hops: int = Field(2, ge=1, le=3)


class KGQAResponse(BaseModel):
    question: str
    answer: str
    hops: int
    confidence: dict[str, Any]
    evidence: list[dict[str, Any]]
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
class NL2SQLRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1_000)


class NL2SQLResponse(BaseModel):
    question: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    generation_mode: str
    error: str = ""
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadyCheck(BaseModel):
    name: str
    ready: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: list[ReadyCheck]


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    request_id: str = "-"
