from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class DataProblemSchema(BaseModel):
    data_type: str
    problem_type: str


class TabularMetaDataSchema(DataProblemSchema):
    id_column: Optional[str] = None
    target_columns: Union[str, List[str]]
    drop_columns: Optional[Union[str, List[str]]] = None


class ImageMetaDataSchema(DataProblemSchema):
    target_columns: Union[str, List[str]]


class TextMetaDataSchema(DataProblemSchema):
    pass


class FileSchema(BaseModel):
    train: str
    valid: Optional[str] = None
    output_dir: str


class AlgorithmSchema(BaseModel):
    model: str
    use_gpu: Optional[bool] = False
    use_tpu: Optional[bool] = False
    n_jobs: Optional[int] = -1
    optimizer: Optional[str] = None
    scheduler: Optional[str] = None
    parameters: Dict[str, Union[int, str, float, List[Union[int, str, float]]]]


class ConfigSchema(BaseModel):
    files: FileSchema
    metadata: Union[TabularMetaDataSchema, ImageMetaDataSchema]
    algorithm: AlgorithmSchema
