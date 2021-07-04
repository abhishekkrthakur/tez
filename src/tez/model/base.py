from abc import ABC

from ..metrics.classification import ClassificationMetrics
from ..metrics.regression import RegressionMetrics
from ..schemas.evaluation import EvaluationResponse


class TezModel(ABC):
    def train(self, train_dataset, **kwargs):
        self.m.fit(train_dataset.data[train_dataset.columns], train_dataset.targets, **kwargs)

    def evaluate(self, validation_dataset, **kwargs) -> EvaluationResponse:
        if validation_dataset.metric_name in (
            "binary_classification",
            "multi_class_classification",
            "multilabel_classification",
        ):
            probas = self.m.predict_proba(validation_dataset.data[validation_dataset.columns])
            preds = self.m.predict(validation_dataset.data[validation_dataset.columns])
            metrics = ClassificationMetrics()
            metrics_dict = metrics.calculate(validation_dataset.targets, preds, probas)

        elif validation_dataset.metric_name in ("single_column_regression", "multi_column_regression"):
            probas = None
            preds = self.m.predict(validation_dataset.data[validation_dataset.columns])
            metrics = RegressionMetrics()
            metrics_dict = metrics.calculate(validation_dataset.targets, preds)

        else:
            raise ValueError(f"Unsupported metric type: {validation_dataset.metric_name}")

        return EvaluationResponse(probas=probas, preds=preds, metrics=metrics_dict)
