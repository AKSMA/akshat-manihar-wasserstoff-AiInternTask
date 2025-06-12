from pydantic import BaseModel
from typing import List, Optional

# Request model for user queries
class QueryRequest(BaseModel):
    question: str

# Model representing a citation or document answer snippet
class DocAnswer(BaseModel):
    DocID: str
    Page: Optional[int]
    ChunkIndex: int
    Excerpt: str

# Response model for query results, including answer and citations
class QueryResponse(BaseModel):
    answer: str
    citations: List[DocAnswer]

# Model representing a theme synthesized from document chunks
class Theme(BaseModel):
    title: str
    documents: List[str]
    summary: str

# Response model for synthesized themes
class ThemeResponse(BaseModel):
    themes: List[Theme]