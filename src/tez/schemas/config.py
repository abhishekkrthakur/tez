from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class ConfigSchema(BaseModel):
    # files
    train: str
    valid: Optional[str] = None
    output_dir: str

    # metadata
    data_type: str
    problem_type: str
    target_columns: Union[str, List[str]]
    drop_columns: Union[str, List[str]]

    # algorithm
    model: str
    use_gpu: Optional[bool] = False
    use_tpu: Optional[bool] = False
    n_jobs: Optional[int] = -1
    parameters: Dict[str, Union[str, int, float, List[Union[str, int, float]]]]
