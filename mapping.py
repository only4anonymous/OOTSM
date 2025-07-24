# mappings.py

from object_classes import OBJECT_CLASSES, RELATIONSHIP_CLASSES

# 对象类映射
OBJECT_ID_TO_LABEL = {idx: label for idx, label in enumerate(OBJECT_CLASSES)}
OBJECT_LABEL_TO_ID = {label: idx for idx, label in enumerate(OBJECT_CLASSES)}

# 关系类映射
RELATIONSHIP_ID_TO_LABEL = {idx: label for idx, label in enumerate(RELATIONSHIP_CLASSES)}
RELATIONSHIP_LABEL_TO_ID = {label: idx for idx, label in enumerate(RELATIONSHIP_CLASSES)}

def get_object_label(obj_id):
    return OBJECT_ID_TO_LABEL.get(obj_id, "unknown_object")

def get_relationship_label(rel_id):
    return RELATIONSHIP_ID_TO_LABEL.get(rel_id, "unknown_relation")