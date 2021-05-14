import sys
from dataclasses import dataclass
from logging import log
from typing import Union

import yaml
from loguru import logger

from ..schemas import (
    ALLOWED_DATA_TYPES,
    ALLOWED_KEYS_METADATA,
    DATA_PROBLEM_MAPPING,
    METADATA_SCHEMA_MAPPING,
    AlgorithmSchema,
    ConfigSchema,
    FileSchema,
    ImageMetaDataSchema,
    TabularMetaDataSchema,
    TextMetaDataSchema,
)


VALID_SECTIONS = {
    "algorithm",
    "files",
    "metadata",
}


@dataclass
class TezConfigParser:
    config_file: str

    def __post_init__(self) -> None:
        with open(self.config_file) as yaml_file:
            self.config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.project_name = list(self.config.keys())[0]

    def _parse_metadata(self, metadata: dict) -> Union[TabularMetaDataSchema, TextMetaDataSchema, ImageMetaDataSchema]:
        if "data_type" not in metadata:
            logger.error(f"Please provide `data_type`. One of: {ALLOWED_DATA_TYPES}")
            sys.exit(1)
        elif metadata["data_type"].strip() not in ALLOWED_DATA_TYPES:
            logger.error(f"`data_type` must be one of: {ALLOWED_DATA_TYPES}")
            sys.exit(1)
        else:
            data_type = metadata["data_type"].strip()

        allowed_problem_types = DATA_PROBLEM_MAPPING[data_type]
        if "problem_type" not in metadata:
            logger.error(f"Please provide `data_type`. One of: {ALLOWED_DATA_TYPES}")
            sys.exit(1)
        elif metadata["problem_type"].strip() not in allowed_problem_types:
            logger.error(f"`problem_type` must be one of: {allowed_problem_types} for `data_type`: {data_type}")
            sys.exit(1)
        else:
            problem_type = metadata["problem_type"].strip()

        allowed_keys = ALLOWED_KEYS_METADATA[data_type][problem_type]
        temp_dict = {k: v for k, v in metadata.items() if k in allowed_keys}
        schema = METADATA_SCHEMA_MAPPING[data_type]
        return schema.parse_obj(temp_dict)

    def _parse_files(self, files: dict) -> FileSchema:
        if files["train"].startswith("~") or files["valid"].startswith("~") or files["output_dir"].startswith("~"):
            logger.error("Please provide full path for `files`")
            sys.exit(1)
        return FileSchema.parse_obj(files)

    def _parse_algorithm(self, algorithm: dict) -> AlgorithmSchema:
        return AlgorithmSchema.parse_obj(algorithm)

    def parse(self) -> ConfigSchema:
        provided_sections = set(self.config[self.project_name])
        project_config = self.config[self.project_name]
        logger.info(provided_sections)
        if (
            "files" not in provided_sections
            or "algorithm" not in provided_sections
            or "metadata" not in provided_sections
        ):
            logger.error(f"`files` not found in config. Please provide a `files` section.")
            sys.exit(1)

        if not provided_sections.issubset(VALID_SECTIONS):
            invalid_sections = provided_sections - VALID_SECTIONS
            logger.warning(f"Invalid sections found. They will be ignored: {invalid_sections}")

        files = self._parse_files(project_config["files"])
        metadata = self._parse_metadata(project_config["metadata"])
        algorithm = self._parse_algorithm(project_config["algorithm"])

        if metadata.drop_columns:
            metadata.drop_columns = metadata.drop_columns.split(",")
            metadata.drop_columns = [d_col.strip() for d_col in metadata.drop_columns]

        conf = {"files": files, "metadata": metadata, "algorithm": algorithm}
        return ConfigSchema.parse_obj(conf)
