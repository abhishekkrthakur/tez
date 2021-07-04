import os
import sys
from dataclasses import dataclass
from enum import unique
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from loguru import logger

from tez import datasets

from .datasets import TabularDataset
from .metrics import metrics
from .model import ModelDispatcher, TezModel
from .parser import TezConfigParser, TezParameterParser
from .schemas import AlgorithmSchema, DataProblemSchema, TabularMetaDataSchema


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

    def _train_tabular_model(self, model, train_dataset, validation_dataset) -> None:
        logger.info(f"Training model: {model}")
        model.train(train_dataset)
        evaluation = model.evaluate(validation_dataset)
        logger.info(f"Metrics: {evaluation.metrics}")

    def start(self):
        if isinstance(self.config.metadata, TabularMetaDataSchema):
            training_data_path = self.config.files.train
            logger.info(f"Training data: {training_data_path}")
            validation_data_path = self.config.files.valid
            logger.info(f"Validation data: {validation_data_path}")

            # TODO: automated folds generation
            if validation_data_path is None:
                logger.error("Validation data is required!")
                sys.exit(1)

            if self.config.metadata.drop_columns is not None:
                if isinstance(self.config.metadata.drop_columns, str):
                    self.config.metadata.drop_columns = [self.config.metadata.drop_columns]

            if self.config.metadata.target_columns is not None:
                if isinstance(self.config.metadata.target_columns, str):
                    self.config.metadata.target_columns = [self.config.metadata.target_columns]

            train_dataset = TabularDataset(
                data_path=training_data_path,
                problem_type=self.config.metadata.problem_type,
                id_column=self.config.metadata.id_column,
                drop_columns=self.config.metadata.drop_columns,
                target_name=self.config.metadata.target_columns,
            )

            validation_dataset = TabularDataset(
                data_path=validation_data_path,
                problem_type=self.config.metadata.problem_type,
                id_column=self.config.metadata.id_column,
                drop_columns=self.config.metadata.drop_columns,
                target_name=self.config.metadata.target_columns,
            )

        models = self._get_model_configs()

        if not models:
            logger.error("No models were generated!")
            sys.exit(1)

        for model in models:
            self._train_tabular_model(model, train_dataset, validation_dataset)
