import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Union

import joblib
from loguru import logger

from .datasets import TabularDataset
from .model import ModelDispatcher, TezModel
from .parser import TezConfigParser, TezParameterParser
from .schemas import AlgorithmSchema, DataProblemSchema, TabularMetaDataSchema
from .utils import generate_random_names


@dataclass
class Tez:
    config_file: str

    def __post_init__(self) -> None:
        parser = TezConfigParser(self.config_file)
        self.config = parser.parse()
        self.logging_dir = None
        self.output_dir = None
        self._create_required_dirs()

    def _create_required_dirs(self, force=True) -> None:
        logger.info(f"Creating directory: {self.config.files.output_dir}")
        try:
            self.output_dir = self.config.files.output_dir
            os.makedirs(self.output_dir, exist_ok=force)
        except FileExistsError:
            logger.error("The output folder already exists!")
            sys.exit(1)

        self.logging_dir = os.path.join(self.config.files.output_dir, "logs")
        os.makedirs(self.logging_dir, exist_ok=True)

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
        model_names = generate_random_names(len(models))
        models = dict(zip(model_names, models))
        return models

    def _train_tabular_model(self, model, model_name, train_dataset, validation_dataset) -> None:
        logger.info(f"Training model: {model}")
        model.train(train_dataset)
        evaluation = model.evaluate(validation_dataset)
        logger.info(f"Metrics: {evaluation.metrics}")
        joblib.dump(evaluation, os.path.join(self.output_dir, f"eval-{model_name}.tez"))
        joblib.dump(model, os.path.join(self.output_dir, f"model-{model_name}.tez"))

    def deploy(self) -> None:
        raise NotImplementedError

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
                target_names=self.config.metadata.target_columns,
                is_training=True,
                output_path=self.output_dir,
            )

            validation_dataset = TabularDataset(
                data_path=validation_data_path,
                problem_type=self.config.metadata.problem_type,
                id_column=self.config.metadata.id_column,
                drop_columns=self.config.metadata.drop_columns,
                target_names=self.config.metadata.target_columns,
                is_training=False,
                output_path=self.output_dir,
            )

        else:
            raise Exception("This type of schema is not implemented yet")

        models = self._get_model_configs()

        if not models:
            logger.error("No models were generated!")
            sys.exit(1)

        for model_name, model in models.items():
            self._train_tabular_model(model, model_name, train_dataset, validation_dataset)
