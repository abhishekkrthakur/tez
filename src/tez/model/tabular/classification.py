from dataclasses import dataclass
from typing import Any, Dict, List, Union

from ...schemas import AlgorithmSchema
from ..base import TezModel


@dataclass
class TabularClassification(TezModel):
    requires_scaling: bool = True
    verbose: bool = True
    random_state: bool = 42
    n_jobs: bool = -1
    gpu: bool = False
    has_predict: bool = True
    has_predict_proba: bool = True


@dataclass
class RandomForest(TabularClassification):
    n_estimators: int = 200
    criterion: str = "gini"
    max_depth: int = 7
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    max_features: Union[int, float, str] = "auto"
    class_weight: Union[Dict[int, float], List[Dict[int, float]]] = None

    def __post_init__(self):
        from sklearn.ensemble import RandomForestClassifier

        self.m = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            verbose=self.verbose,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.requires_scaling = False
        self.gpu = False
        self.has_predict = True
        self.has_predict_proba = True


@dataclass
class ExtraTrees(TabularClassification):
    n_estimators: int = 200
    criterion: str = "gini"
    max_depth: int = 7
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    max_features: Union[int, float, str] = "auto"
    class_weight: Union[Dict[int, float], List[Dict[int, float]]] = None

    def __post_init__(self):
        from sklearn.ensemble import ExtraTreesClassifier

        self.m = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            verbose=self.verbose,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.requires_scaling = False
        self.gpu = False
        self.has_predict = True
        self.has_predict_proba = True
