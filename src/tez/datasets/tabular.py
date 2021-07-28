import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base import TezDataset


@dataclass
class TabularDataset(TezDataset):
    data_path: str
    problem_type: str
    is_training: bool
    id_column: Optional[str]
    output_path: str
    target_names: Optional[List[str]] = None
    separator: Optional[str] = ","
    drop_columns: Optional[List[str]] = None
    column_types: Optional[Dict[str, str]] = None

    @property
    def columns(self):
        return [k for k in self.data_meta["column_types"].keys()]

    def __post_init__(self):
        self.data = pd.read_csv(self.data_path, sep=self.separator, dtype=self.column_types)
        data_meta_path = os.path.join(self.output_path, "data_meta.tez")

        if self.is_training:
            if self.drop_columns:
                self.data = self.data.drop(self.drop_columns, axis=1)

            if self.id_column:
                self.data = self.data.drop(self.id_column, axis=1)

            column_types = self.data.dtypes.to_dict()
            column_types = {k: v.name for k, v in column_types.items() if k not in self.target_names}
            self.column_types = column_types

            self.label_encoder = {}
            self.one_hot_encoder = {}
            self.target_encoder = {}

            # encode column_types with label encoder if type is object type
            for col_name, col_type in self.column_types.items():
                if col_type == "object":
                    lbl_enc = LabelEncoder()
                    lbl_enc.fit(self.data[col_name])
                    self.label_encoder[col_name] = lbl_enc
                    self.data[col_name] = lbl_enc.transform(self.data[col_name])

            self.num_targets = len(self.target_names)
            self.targets = self.data[self.target_names]

            if self.num_targets == 1:
                self.unique_targets = len(np.unique(self.targets))
                if self.unique_targets == 2 and self.problem_type == "classification":
                    self.metric_name = "binary_classification"
                elif self.unique_targets > 2 and self.problem_type == "classification":
                    self.metric_name = "multi_class_classification"
                else:
                    self.metric_name = "single_column_regression"
            else:
                self.unique_targets = None
                if self.problem_type == "classification":
                    self.metric_name = "multi_label_classification"
                else:
                    self.metric_name = "multi_column_regression"

            if self.metric_name in (
                "binary_classification",
                "multi_class_classification",
                "multi_label_classification",
            ):
                for col_name in self.target_names:
                    lbl_enc = LabelEncoder()
                    lbl_enc.fit(self.data[col_name])
                    self.target_encoder[col_name] = lbl_enc
                    self.data[col_name] = lbl_enc.transform(self.data[col_name])

            self.targets = self.data[self.target_names]
            if self.num_targets == 1:
                self.targets = self.targets.values.ravel()

            self.data_meta = {
                "data_type": "tabular",
                "column_types": self.column_types,
                "label_encoder": self.label_encoder,
                "one_hot_encoder": self.one_hot_encoder,
                "drop_columns": self.drop_columns,
                "target_columns": self.target_names,
                "metric_name": self.metric_name,
                "id_column": self.id_column,
                "unique_targets": self.unique_targets,
                "target_encoder": self.target_encoder,
            }
            joblib.dump(self.data_meta, data_meta_path)

        else:
            self.data_meta = joblib.load(data_meta_path)
            self.column_types = self.data_meta["column_types"]
            self.label_encoder = self.data_meta["label_encoder"]
            self.one_hot_encoder = self.data_meta["one_hot_encoder"]
            self.drop_columns = self.data_meta["drop_columns"]
            self.target_names = self.data_meta["target_columns"]
            self.metric_name = self.data_meta["metric_name"]
            self.id_column = self.data_meta["id_column"]
            self.unique_targets = self.data_meta["unique_targets"]
            self.target_encoder = self.data_meta["target_encoder"]

            # encode column_types with label encoder
            for col_name, col_type in self.column_types.items():
                if col_name in self.label_encoder:
                    self.data[col_name] = self.label_encoder[col_name].transform(self.data[col_name])

            # encode target_names with label encoder
            for col_name in self.target_names:
                if col_name in self.target_encoder:
                    self.data[col_name] = self.target_encoder[col_name].transform(self.data[col_name])

            if self.target_names[0] in self.data.columns:
                self.targets = self.data[self.target_names]
            else:
                self.targets = None
