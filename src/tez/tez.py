import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Union

from loguru import logger

from .model import ModelDispatcher, TezModel
from .parser import TezConfigParser, TezParameterParser
from .schemas import AlgorithmSchema, DataProblemSchema


@dataclass
class Tez:
    config_file: str

    def __post_init__(self) -> None:
        parser = TezConfigParser(self.config_file)
        self.config = parser.parse()
        self._create_required_dirs()

    def _create_required_dirs(self, force=True) -> None:
        logger.info(f"Creating directory: {self.config.files.output_dir}")
        try:
            os.makedirs(self.config.files.output_dir, exist_ok=force)
        except FileExistsError:
            logger.error("The output folder already exists!")
            sys.exit(1)

        logging_dir = os.path.join(self.config.files.output_dir, "logs")
        os.makedirs(logging_dir, exist_ok=True)

    def _create_params_list(
        self, algorithm: AlgorithmSchema
    ) -> List[Dict[str, Union[int, str, float, List[Union[int, str, float]]]]]:
        parser = TezParameterParser(algorithm.parameters)
        params = parser.parse_params()
        return params

    def _get_model_configs(self) -> List[TezModel]:
        data_problem = DataProblemSchema(
            data_type=self.config.metadata.data_type, problem_type=self.config.metadata.problem_type
        )
        param_list = self._create_params_list(self.config.algorithm)
        model_dispatcher = ModelDispatcher(algo=self.config.algorithm, data_problem=data_problem)
        models = [model_dispatcher.dispatch(parameters=plist) for plist in param_list]
        return models

    def start(self):
        models = self._get_model_configs()
        print(models)
