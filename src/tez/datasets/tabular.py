from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import TezDataset


@dataclass
class TabularDataset(TezDataset):
    data_path: str
    problem_type: str
    target_name: List[str]
    id_column: Optional[str]
    separator: Optional[str] = ","
    drop_columns: Optional[List[str]] = None
    column_types: Optional[Dict[str, str]] = None

    @property
    def metric_name(self):
        if self.num_targets == 1:
            unique_targets = len(np.unique(self.targets))
            if unique_targets == 2 and self.problem_type == "classification":
                metric_name = "binary_classification"
            elif unique_targets > 2 and self.problem_type == "classification":
                metric_name = "multi_class_classification"
            else:
                metric_name = "single_column_regression"
        else:
            if self.problem_type == "classification":
                metric_name = "multi_label_classification"
            else:
                metric_name = "multi_column_regression"
        return metric_name

    def __post_init__(self):
        self.data = pd.read_csv(self.data_path, sep=self.separator, dtype=self.column_types)
        if self.drop_columns:
            self.data = self.data.drop(self.drop_columns, axis=1)

        if self.id_column:
            self.data = self.data.drop(self.id_column, axis=1)

        column_types = self.data.dtypes.to_dict()
        column_types = {k: v.name for k, v in column_types.items()}
        self.column_types = column_types

        self.targets = self.data[self.target_name]
        self.data = self.data.drop(self.target_name, axis=1)

        self.num_targets = len(self.target_name)

        if self.num_targets == 1:
            self.targets = self.targets.values.ravel()
