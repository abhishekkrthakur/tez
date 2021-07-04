from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel


class NumpyNDArray(BaseModel):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> str:
        # validate data...
        return v


class EvaluationResponse(BaseModel):
    probas: Optional[NumpyNDArray] = None
    preds: NumpyNDArray
    metrics: Dict[str, float]
