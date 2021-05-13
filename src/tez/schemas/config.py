from typing import Optional

from pydantic import BaseModel


class ConfigSchema(BaseModel):
    train: str
    valid: Optional[str] = None
    output_dir: str
