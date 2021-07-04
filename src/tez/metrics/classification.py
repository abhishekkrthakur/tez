import numba
import numpy as np
from sklearn import metrics as skmetrics


class ClassificationMetrics:
    def __init__(self):
        """
        init class for classification metrics
        """
        self.metrics = {
            "accuracy": self._accuracy,
            "auc": self._auc,
            "f1": self._f1,
            "kappa": self._kappa,
            "logloss": self._log_loss,
            "multiclass_logloss": self._multiclass_log_loss,
            "precision": self._precision,
            "quadratic_kappa": self._quadratic_weighted_kappa,
            "recall": self._recall,
        }

    def calculate(self, y_test, y_pred, y_proba):
        """
        calculate metrics for classification
        :param y_test: true labels
        :param y_pred: predicted labels
        :param y_proba: predicted probabilities
        :return: metrics dictionary
        """
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = self.metrics[metric](y_test, y_pred, y_proba)
        return metrics

    def __call__(self, metric, y_test, y_pred, y_proba):
        if metric not in self.metrics:
            raise Exception("Invalid metric passed")
        if metric == "auc":
            if y_proba is not None:
                return self.metrics[metric](y_test, y_proba[:, 1])
            else:
                return np.nan
        elif metric == "logloss":
            if y_proba is not None:
                return self.metrics[metric](y_test, y_proba[:, 1])
            else:
                return np.nan
        elif metric == "multiclass_logloss":
            if y_proba is not None:
                return self.metrics[metric](y_test, y_proba)
            else:
                return np.nan
        else:
            return self.metrics[metric](y_test, y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    def _auc(self, y_true, y_pred, fast=False):
        if fast is False:
            return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        else:
            return self._fast_auc(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    @numba.jit
    def _fast_auc(y_true, y_pred):
        """
        This function has been taken from: https://github.com/jfpuget/metrics/blob/master/auc.ipynb
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_pred)]
        n_false = 0
        auc = 0
        n = len(y_true)
        for i in range(n):
            y_i = y_true[i]
            n_false += 1 - y_i
            auc += y_i * n_false
        auc /= n_false * (n - n_false)
        return auc

    @staticmethod
    def _log_loss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _kappa(y1, y2, weights):
        return skmetrics.cohen_kappa_score(y1, y2, weights)

    @staticmethod
    def _multiclass_log_loss(y_true, y_pred, eps=1e-15):
        """
        Multi class version of Logarithmic Loss metric.
        Taken from: https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/discussion/2644#14250
        :param y_true : array, shape = [n_samples]
        :param y_pred : array, shape = [n_samples, n_classes]
        :return loss : float
        """
        predictions = np.clip(y_pred, eps, 1 - eps)

        # normalize row sums to 1
        predictions /= predictions.sum(axis=1)[:, np.newaxis]

        actual = np.zeros(y_pred.shape)
        rows = actual.shape[0]
        actual[np.arange(rows), y_true.astype(int)] = 1
        vsota = np.sum(actual * np.log(predictions))
        return -1.0 / rows * vsota

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _quadratic_weighted_kappa(y1, y2):
        return skmetrics.cohen_kappa_score(y1, y2, weights="quadratic")

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
