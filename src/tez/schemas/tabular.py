from typing import List, Union

from pydantic import BaseModel


class TabularDataSchema(BaseModel):
    target_columns: Union[str, List[str]]
    drop_columns: Union[str, List[str]]


class TabularClassifcationDataSchema(TabularDataSchema):
    pass


class TabularRegressionDataSchema(TabularDataSchema):
    pass
