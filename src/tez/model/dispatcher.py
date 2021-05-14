import importlib
from dataclasses import dataclass

from ..schemas import AlgorithmSchema, DataProblemSchema


MODEL_NAME_MAPPING = {
    "randomforest": "RandomForest",
}


@dataclass
class ModelDispatcher:
    algo: AlgorithmSchema
    data_problem: DataProblemSchema

    def __post_init__(self):
        self.model_name = self.algo.model
        self.data_type = self.data_problem.data_type
        self.problem_type = self.data_problem.problem_type
        self.model_class_name = MODEL_NAME_MAPPING[self.model_name]

    def dispatch(self, parameters: dict):
        _model = importlib.import_module(f".model.{self.data_type}.{self.problem_type}", package="tez")
        model = getattr(_model, self.model_class_name)
        return model(**parameters)
