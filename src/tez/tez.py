import os
import sys
from dataclasses import dataclass

from loguru import logger

from .config import TezParser


@dataclass
class Tez:
    config_file: str

    def __post_init__(self):
        tez_parser = TezParser(self.config_file)
        self.parsed_config = tez_parser.parse()
        self._create_required_dirs()

    def _create_required_dirs(self, force=False):
        logger.info(f"Creating directory: {self.parsed_config.output_dir}")
        try:
            os.makedirs(self.parsed_config.output_dir, exist_ok=force)
        except FileExistsError:
            logger.error("The output folder already exists!")
            sys.exit(1)

        logging_dir = os.path.join(self.parsed_config.output_dir, "logs")
        os.makedirs(logging_dir)

    def start(self):
        pass
