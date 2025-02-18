from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5  # Default to returning top 5 results

class AskQuery(BaseModel):
    query: str

class FileUploadRequest(BaseModel):
    file_names: List[str]

class SearchResponse(BaseModel):
    query: str
    results: List[dict]

class AnswerResponse(BaseModel):
    query: str
    answer: str
