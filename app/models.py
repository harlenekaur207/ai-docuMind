from typing import List

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    reference: int
    filename: str
    page: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

