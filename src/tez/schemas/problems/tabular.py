from typing import List, Union

from pydantic import BaseModel


class TabularClassifcationSchema:
    target_columns: Union[str, List[str]]
    drop_columns: Union[str, List[str]]


class TabularRegressionSchema:
    target_columns: Union[str, List[str]]
    drop_columns: Union[str, List[str]]
