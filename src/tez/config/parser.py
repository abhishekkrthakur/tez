import sys
from dataclasses import dataclass
from logging import log

import yaml
from loguru import logger

from ..schemas import ALLOWED_DATA_TYPES, ConfigSchema


VALID_SECTIONS = {
    "algorithm",
    "files",
    "metadata",
}


@dataclass
class TezParser:
    config_file: str

    def __post_init__(self) -> None:
        with open(self.config_file) as yaml_file:
            self.config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.project_name = list(self.config.keys())[0]

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

        files = project_config["files"]
        metadata = project_config["metadata"]
        algorithm = project_config["algorithm"]

        if "data_type" not in metadata:
            logger.error(f"Please provide `data_type`. One of: {ALLOWED_DATA_TYPES}")
            sys.exit(1)
        elif metadata["data_type"].strip() not in ALLOWED_DATA_TYPES:
            logger.error(f"`data_type` must be one of: {ALLOWED_DATA_TYPES}")
            sys.exit(1)
        else:
            data_type = metadata["data_type"].strip()

        # TODO: add a lot of checks

        if (
            project_config["files"]["train"].startswith("~")
            or project_config["files"]["valid"].startswith("~")
            or project_config["files"]["output_dir"].startswith("~")
        ):
            logger.error("Please provide full path for `files`")
            sys.exit(1)

        if project_config["files"].get("drop_columns") is None:
            drop_columns = None
        else:
            drop_columns = project_config["files"].get("drop_columns")
            drop_columns = drop_columns.split(",")
            drop_columns = [d_col.strip() for d_col in drop_columns]

        conf = {
            "train": project_config["files"]["train"],
            "valid": project_config["files"]["valid"],
            "output_dir": project_config["files"]["output_dir"],
            "problem_type": project_config["files"]["problem_type"],
            "target_columns": project_config["files"]["target_columns"],
            "drop_columns": drop_columns,
        }
        return ConfigSchema.parse_obj(conf)
