from .config import ImageMetaDataSchema, TabularMetaDataSchema, TextMetaDataSchema


ALLOWED_DATA_TYPES = [
    "tabular",
    "image",
    "text",
]

ALLOWED_TABULAR_PROBLEM_TYPES = [
    "classification",
    "regression",
]

ALLOWED_IMAGE_PROBLEM_TYPES = [
    "classification",
    "regression",
    "instance_segmentation",
    "semantic_segmentation",
    "object_detection",
]

ALLOWED_TEXT_PROBLEM_TYPES = [
    "classification",
    "regression",
    "entity_extraction",
    "question_answering",
    "summarization",
]

DATA_PROBLEM_MAPPING = {
    "tabular": ALLOWED_TABULAR_PROBLEM_TYPES,
    "image": ALLOWED_IMAGE_PROBLEM_TYPES,
    "text": ALLOWED_TEXT_PROBLEM_TYPES,
}

ALLOWED_KEYS_METADATA = {
    "tabular": {
        "classification": [
            "data_type",
            "problem_type",
            "target_columns",
            "drop_columns",
        ],
        "regression": [
            "data_type",
            "problem_type",
            "target_columns",
            "drop_columns",
        ],
    },
    "image": {},
    "text": {},
}

METADATA_SCHEMA_MAPPING = {
    "tabular": TabularMetaDataSchema,
    "image": ImageMetaDataSchema,
    "text": TextMetaDataSchema,
}
