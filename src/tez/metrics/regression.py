import numpy as np
from sklearn import metrics as skmetrics


class RegressionMetrics:
    def __init__(self):
        """
        init class for regression metrics
        """
        self.metrics = {
            "mae": self._mae,
            "msle": self._msle,
            "mse": self._mse,
            "rmsle": self._rmsle,
            "rmse": self._rmse,
            "r2": self._r2,
        }

    def calculate(self, y_test, y_pred):
        """
        calculate metrics for classification
        :param y_test: true labels
        :param y_pred: predictions
        :return: metrics dictionary
        """
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = self.metrics[metric](y_test, y_pred)
        return metrics

    def __call__(self, metric, y_test, y_pred):
        if metric not in self.metrics:
            raise Exception("Invalid metric passed")
        return self.metrics[metric](y_test, y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _msle(y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true=y_true, y_pred=y_pred))

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def _r2(y_true, y_pred):
        return skmetrics.r2_score(y_true=y_true, y_pred=y_pred)
