from typing import Dict, List, Optional

import pandas as pd

from ..base import TezDataset


class TabularClassificationDataset(TezDataset):
    data_path: str
    data_type: str
    features: List[str]
    targets: List[str]
    id_column: str
    separator: Optional[str] = ","
    drop_columns: Optional[List[str]] = None
    column_types: Optional[Dict[str, str]] = None

    def __post_init__(self):
        self.df = pd.read_csv(self.data_path, sep=self.separator, dtype=self.column_types)
        if self.drop_columns:
            self.df = self.df.drop(self.drop_columns, axis=1)
        column_types = self.df.dtypes.to_dict()
        column_types = {k: v.name for k, v in column_types.items()}
        self.column_types = column_types
