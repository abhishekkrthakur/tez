import sys
from dataclasses import dataclass
from logging import log

import yaml
from loguru import logger

from ..schemas import ConfigSchema


VALID_SECTIONS = {
    "algorithm",
    "files",
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
        if "files" not in provided_sections:
            logger.error(f"`files` not found in config. Please provide a `files` section.")
            sys.exit(1)

        if not provided_sections.issubset(VALID_SECTIONS):
            invalid_sections = provided_sections - VALID_SECTIONS
            logger.warning(f"Invalid sections found. They will be ignored: {invalid_sections}")

        valid_sections = provided_sections.intersection(VALID_SECTIONS)
        if (
            project_config["files"]["train"].startswith("~")
            or project_config["files"]["valid"].startswith("~")
            or project_config["files"]["output_dir"].startswith("~")
        ):
            logger.error("Please provide full path for `files`")
            sys.exit(1)

        conf = {
            "train": project_config["files"]["train"],
            "valid": project_config["files"]["valid"],
            "output_dir": project_config["files"]["output_dir"],
        }
        return ConfigSchema.parse_obj(conf)
