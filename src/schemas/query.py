from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    image_id: Optional[str] = None
    image_id_t2: Optional[str] = None
    image_id_s1: Optional[str] = None
    image_id_s2: Optional[str] = None
    bigearthnet_patch_id: Optional[str] = None
    modality: Optional[str] = None
