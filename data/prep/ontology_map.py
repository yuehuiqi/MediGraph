"""Mappings from CMeIE-V2 / DiaKG / CM3KG label spaces to the MediGraph ontology.

Only the relations/entity-types that have a faithful equivalent in our ontology
(medigraph/schema/ontology.py) are kept, so the resulting gold set yields an
honest F1 on real benchmark data.
"""

# ---- CM3KG (medical.csv columns) -> (relation, tail_type). Head is the disease. ----
CM3KG_COLS = {
    "symptom": ("has_symptom", "Symptom"),
    "recommand_drug": ("recommend_drug", "Drug"),
    "common_drug": ("recommend_drug", "Drug"),
    "check": ("need_examination", "Examination"),
    "cure_department": ("treated_in_department", "Department"),
    "acompany": ("complication", "Disease"),
}

# ---- DiaKG entity_type -> our entity type (others dropped) ----
DIAKG_ENT = {
    "Disease": "Disease",
    "Drug": "Drug",
    "Symptom": "Symptom",
    "Test": "Examination",
    "Test_items": "Examination",
    "Anatomy": "Body",
    "Treatment": "Procedure",   # 非药物治疗手段(饮食/运动/手术等)
    "Operation": "Procedure",   # 手术
    "ADE": "Symptom",           # 药物不良事件，表现为症状
}
# DiaKG relation_type -> our relation. DiaKG names are "<Head>_<Tail>"; build_diakg
# canonicalizes so the *tail* entity is the subject of our relation (so Disease
# heads disease-centric relations; Drug heads adverse_reaction).
DIAKG_REL = {
    "Symptom_Disease": "has_symptom",
    "Drug_Disease": "recommend_drug",
    "Test_Disease": "need_examination",
    "Test_items_Disease": "need_examination",
    "Anatomy_Disease": "located_in",
    "Treatment_Disease": "treated_by_procedure",
    "Operation_Disease": "treated_by_procedure",
    "ADE_Drug": "adverse_reaction",
}

# ---- CMeIE-V2 entity types (Chinese) -> our entity type ----
CMEIE_ENT = {
    "疾病": "Disease",
    "症状": "Symptom",
    "检查": "Examination",
    "药物": "Drug",
    "部位": "Body",
}
# CMeIE-V2 predicate (Chinese) -> our relation. Subject is 疾病 for these.
CMEIE_REL = {
    "临床表现": "has_symptom",
    "相关（症状）": "has_symptom",
    "药物治疗": "recommend_drug",
    "辅助检查": "need_examination",
    "影像学检查": "need_examination",
    "实验室检查": "need_examination",
    "组织学检查": "need_examination",
    "内窥镜检查": "need_examination",
    "筛查": "need_examination",
    "就诊科室": "treated_in_department",
    "并发症": "complication",
    "发病部位": "located_in",
    "转移部位": "located_in",
    "外侵部位": "located_in",
    "手术治疗": "treated_by_procedure",
    "放射治疗": "treated_by_procedure",
    "化疗": "treated_by_procedure",
}
# tail entity type to assign when CMeIE object_type is "其他"/unmapped (fallback)
CMEIE_REL_TAIL_TYPE = {
    "treated_in_department": "Department",
    "treated_by_procedure": "Procedure",
    "need_examination": "Examination",
    "located_in": "Body",
    "has_symptom": "Symptom",
    "recommend_drug": "Drug",
    "complication": "Disease",
}
