# -*- coding: utf-8 -*-
"""Held-out generalisation probe for the NL2SQL router.

The three reported eval sets are all either template-generated or were used to
patch `_router_should_defer` until it deferred exactly the questions it got
wrong. That makes their 100% a *development* number. These questions were
written fresh against the DB schema and appear in none of the three sets, so
they measure what the router does on inputs it was never tuned against.

Offline only: the deterministic path costs nothing. Questions the router hands
to the LLM are reported separately and not scored here.

Usage:
  python benchmarks/eval_nl2sql_holdout.py
Writes outputs/eval_nl2sql_holdout.json.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.analysis.nl2sql import NL2SQL  # noqa: E402

DB = str(OUTPUTS_DIR / "analytics.db")

# (question, gold SQL) -- all hand-written, none present in gold/stress/hard sets.
CASES = [
    ("儿科的平均就诊费用是多少",
     "SELECT AVG(cost) FROM patient_visits WHERE department='儿科'"),
    ("脂肪肝患者的平均年龄是多少",
     "SELECT AVG(age) FROM patient_visits WHERE disease='脂肪肝'"),
    ("肿瘤科有多少次就诊",
     "SELECT COUNT(*) FROM patient_visits WHERE department='肿瘤科'"),
    ("贫血患者有多少人次就诊",
     "SELECT COUNT(*) FROM patient_visits WHERE disease='贫血'"),
    ("45岁以上的患者有多少次就诊",
     "SELECT COUNT(*) FROM patient_visits WHERE age>45"),
    ("男性患者一共有多少次就诊",
     "SELECT COUNT(*) FROM patient_visits WHERE gender='男'"),
    ("女性脂肪肝患者有多少次就诊",
     "SELECT COUNT(*) FROM patient_visits WHERE gender='女' AND disease='脂肪肝'"),
    ("各科室的平均就诊年龄是多少",
     "SELECT department, AVG(age) AS a FROM patient_visits GROUP BY department ORDER BY a DESC"),
    ("就诊量最少的科室是哪个",
     "SELECT department, COUNT(*) AS c FROM patient_visits GROUP BY department ORDER BY c ASC LIMIT 1"),
    ("平均就诊费用最高的科室是哪个",
     "SELECT department, AVG(cost) AS a FROM patient_visits GROUP BY department ORDER BY a DESC LIMIT 1"),
    ("就诊人次最高的5种疾病",
     "SELECT disease, COUNT(*) AS c FROM patient_visits GROUP BY disease ORDER BY c DESC LIMIT 5"),
    ("开具次数最多的3种药物",
     "SELECT drug, COUNT(*) AS c FROM prescriptions GROUP BY drug ORDER BY c DESC LIMIT 3"),
    ("清凉油一共被开具了多少次",
     "SELECT COUNT(*) FROM prescriptions WHERE drug='清凉油'"),
    ("跌打丸开了多少次",
     "SELECT COUNT(*) FROM prescriptions WHERE drug='跌打丸'"),
    ("每个月分别有多少次就诊",
     "SELECT substr(visit_date,1,7) AS m, COUNT(*) AS v FROM patient_visits GROUP BY m ORDER BY m"),
    ("所有就诊的费用总计是多少",
     "SELECT SUM(cost) FROM patient_visits"),
    ("血常规的异常次数是多少",
     "SELECT SUM(abnormal) FROM lab_tests WHERE test_name='血常规'"),
    ("各检查项目的异常次数是多少",
     "SELECT test_name, SUM(abnormal) AS a FROM lab_tests GROUP BY test_name ORDER BY a DESC"),
    ("异常率最高的检查项目是哪个",
     "SELECT test_name, AVG(abnormal) AS r FROM lab_tests GROUP BY test_name ORDER BY r DESC"),
    ("按性别统计就诊人次",
     "SELECT gender, COUNT(*) AS c FROM patient_visits GROUP BY gender ORDER BY c DESC"),
    ("每种疾病的平均就诊费用是多少",
     "SELECT disease, AVG(cost) AS a FROM patient_visits GROUP BY disease ORDER BY a DESC"),
    ("肝病科接诊了多少人次",
     "SELECT COUNT(*) FROM patient_visits WHERE department='肝病'"),
    ("20到40岁的患者有多少次就诊",
     "SELECT COUNT(*) FROM patient_visits WHERE age BETWEEN 20 AND 40"),
    ("高血脂患者的平均住院费用是多少",
     "SELECT AVG(cost) FROM patient_visits WHERE disease='高血脂'"),
    ("外科的平均就诊年龄是多少",
     "SELECT AVG(age) FROM patient_visits WHERE department='外科'"),
    ("费用最高的3次就诊是哪些患者",
     "SELECT patient_id, cost FROM patient_visits ORDER BY cost DESC LIMIT 3"),
]


def ms(rows):
    return Counter(tuple(round(x, 2) if isinstance(x, float) else x for x in r) for r in rows)


def run(sql):
    try:
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        r = c.execute(sql).fetchall()
        c.close()
        return True, r
    except Exception as e:  # noqa: BLE001
        return False, [str(e)]


def match(gold, pred):
    if ms(gold) == ms(pred):
        return True
    gw = len(gold[0]) if gold else 0
    pw = len(pred[0]) if pred else 0
    if gold and pred and pw > gw:
        return ms(gold) == ms([r[:gw] for r in pred])
    return False


def main():
    eng = NL2SQL(DB, llm=object())          # llm never touched on the template path
    hit = ok = deferred = 0
    details = []
    fails = []
    for i, (q, gold_sql) in enumerate(CASES, 1):
        sql, _ = eng._deterministic_sql(q)
        if not sql:
            deferred += 1
            details.append({"question": q, "routed": "llm", "gold_sql": gold_sql})
            print(f"{i:3d}. -- DEFER  {q}")
            continue
        hit += 1
        gok, grows = run(gold_sql)
        pok, prows = run(sql)
        good = gok and pok and match(grows, prows)
        ok += int(good)
        details.append({"question": q, "routed": "deterministic_template",
                        "match": good, "gold_sql": gold_sql, "pred_sql": sql})
        print(f"{i:3d}. {'OK  ' if good else 'FAIL'}  {q}")
        if not good:
            print(f"       pred: {sql}")
            print(f"       gold: {gold_sql}")
            print(f"       pred_rows={prows[:3]}  gold_rows={grows[:3]}")
            fails.append(q)
    report = {
        "description": (
            "Held-out probe: questions written fresh, present in none of "
            "nl2sql_gold.json / nl2sql_stress_128.json / nl2sql_hard_natural.json, "
            "and never used to tune the router. Deterministic path only (no LLM call)."
        ),
        "samples": len(CASES),
        "router_answered": hit,
        "deferred_to_llm": deferred,
        "correct": ok,
        "router_accuracy": round(ok / hit, 4) if hit else 0.0,
        "failures": fails,
        "details": details,
    }
    out = OUTPUTS_DIR / "eval_nl2sql_holdout.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n================ HELD-OUT ROUTER PROBE ================")
    print(f"  total questions : {len(CASES)}")
    print(f"  router answered : {hit}   (deferred to LLM: {deferred})")
    if hit:
        print(f"  correct         : {ok}/{hit} = {ok/hit:.1%}")
    if fails:
        print("  failures:")
        for f in fails:
            print("   -", f)
    print(f"  (saved -> {out})")


if __name__ == "__main__":
    main()
