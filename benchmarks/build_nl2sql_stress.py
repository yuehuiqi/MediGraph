"""Generate a deterministic 128-question compositional NL2SQL stress set.

This set is explicitly labelled template-generated.  It expands breadth and
failure analysis; the original 16-question hand-written set remains reported
separately and is never replaced by this easier reproducibility set.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "nl2sql_stress_128.json"


def main() -> None:
    samples: list[dict] = []

    def add(question: str, sql: str, category: str) -> None:
        samples.append({"question": question, "sql": sql, "category": category})

    total_templates = [
        "总共有多少次就诊", "全部就诊记录一共有多少人次", "就诊总数是多少",
        "请统计就诊次数", "所有患者合计多少次就诊", "累计有多少人次就诊",
        "数据库中的就诊总数", "总共记录了多少次就诊",
    ]
    for question in total_templates:
        add(question, "SELECT COUNT(*) FROM patient_visits", "scalar_count")

    for age in (30, 40, 50, 60, 70):
        for word, operator in (("以上", ">"), ("以下", "<")):
            add(
                f"{age}岁{word}的患者有多少人次",
                f"SELECT COUNT(*) FROM patient_visits WHERE age{operator}{age}",
                "age_filter",
            )
            add(
                f"年龄{age}岁{word}共有多少次就诊",
                f"SELECT COUNT(*) FROM patient_visits WHERE age{operator}{age}",
                "age_filter_paraphrase",
            )

    for gender, value in (("女性", "女"), ("男性", "男")):
        for disease in ("高血压", "糖尿病足", "冠心病", "老年痴呆"):
            add(
                f"{gender}{disease}患者有多少次就诊",
                f"SELECT COUNT(*) FROM patient_visits WHERE gender='{value}' AND disease='{disease}'",
                "multi_condition",
            )

    group_specs = (
        ("科室", "department"), ("疾病", "disease"), ("性别", "gender"),
    )
    for label, column in group_specs:
        for wording in (
            f"每个{label}的就诊量是多少",
            f"各{label}分别有多少人次",
            f"按{label}统计就诊次数",
            f"{label}就诊数量排名",
        ):
            add(
                wording,
                f"SELECT {column}, COUNT(*) AS cnt FROM patient_visits GROUP BY {column} ORDER BY cnt DESC",
                f"group_count_{column}",
            )
        add(
            f"每个{label}的平均就诊费用是多少",
            f"SELECT {column}, AVG(cost) AS avg_cost FROM patient_visits GROUP BY {column} ORDER BY avg_cost DESC",
            f"group_avg_cost_{column}",
        )
        add(
            f"各{label}患者的平均年龄",
            f"SELECT {column}, AVG(age) AS avg_age FROM patient_visits GROUP BY {column} ORDER BY avg_age DESC",
            f"group_avg_age_{column}",
        )

    for n in range(1, 11):
        add(
            f"就诊量最高的{n}个科室",
            f"SELECT department, COUNT(*) AS cnt FROM patient_visits GROUP BY department ORDER BY cnt DESC LIMIT {n}",
            "top_department",
        )
        add(
            f"开具次数最多的{n}种药物",
            f"SELECT drug, COUNT(*) AS cnt FROM prescriptions GROUP BY drug ORDER BY cnt DESC LIMIT {n}",
            "top_drug",
        )
        add(
            f"就诊人次最高的{n}种疾病",
            f"SELECT disease, COUNT(*) AS cnt FROM patient_visits GROUP BY disease ORDER BY cnt DESC LIMIT {n}",
            "top_disease",
        )

    for wording in (
        "2024年每个月的就诊量分别是多少", "月度就诊人次趋势",
        "每月就诊数量变化", "按月份统计就诊量",
        "每个月有多少次就诊", "就诊量的月度趋势",
        "各月患者人次", "年度内每月就诊量",
    ):
        add(
            wording,
            "SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visits "
            "FROM patient_visits GROUP BY month ORDER BY month",
            "monthly_trend",
        )

    for wording in (
        "每个检查项目的异常次数是多少", "按检查项目统计异常数量",
        "各检查的异常结果次数", "检查项目异常数排名",
        "所有检验项目分别有多少次异常",
    ):
        add(
            wording,
            "SELECT test_name, SUM(abnormal) AS abnormal_count FROM lab_tests "
            "GROUP BY test_name ORDER BY abnormal_count DESC",
            "lab_abnormal_count",
        )
    for wording in (
        "检查结果异常率最高的检查项目", "各检查项目的异常比例",
        "按检验项目统计异常率", "检查异常率排名", "每项检查的异常率是多少",
    ):
        add(
            wording,
            "SELECT test_name, AVG(abnormal) AS abnormal_rate FROM lab_tests "
            "GROUP BY test_name ORDER BY abnormal_rate DESC",
            "lab_abnormal_rate",
        )

    for question in (
        "所有就诊的总费用是多少", "全部患者费用总计", "费用总计是多少",
        "累计就诊总费用", "请计算所有记录的总费用", "患者总费用共计多少",
    ):
        add(question, "SELECT SUM(cost) AS total_cost FROM patient_visits", "total_cost")

    for department in ("内科", "外科", "儿科", "肝病"):
        add(
            f"{department}的平均就诊费用是多少",
            f"SELECT AVG(cost) FROM patient_visits WHERE department='{department}'",
            "department_avg_cost",
        )
    for disease in ("高血压", "糖尿病足", "急性心肌梗死", "冠心病"):
        add(
            f"{disease}患者的平均年龄是多少",
            f"SELECT AVG(age) FROM patient_visits WHERE disease='{disease}'",
            "disease_avg_age",
        )
        add(
            f"{disease}患者有多少次就诊",
            f"SELECT COUNT(*) FROM patient_visits WHERE disease='{disease}'",
            "disease_count",
        )
        add(
            f"诊断为{disease}的就诊人次是多少",
            f"SELECT COUNT(*) FROM patient_visits WHERE disease='{disease}'",
            "disease_count_paraphrase",
        )
    for drug in ("阿司匹林肠溶片", "苯磺酸氨氯地平胶囊", "硝苯地平缓释片Ⅰ", "胰岛素注射液"):
        add(
            f"{drug}一共被开具了多少次",
            f"SELECT COUNT(*) FROM prescriptions WHERE drug='{drug}'",
            "drug_count",
        )

    # Keep the contract stable even when categories above are extended later.
    samples = samples[:128]
    assert len(samples) == 128
    payload = {
        "description": (
            "128-question deterministic compositional stress set. Template-generated "
            "for reproducibility and coverage; report separately from the hand-written gold set."
        ),
        "samples": samples,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples -> {OUT}")


if __name__ == "__main__":
    main()
