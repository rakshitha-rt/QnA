from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    type: str = Field(..., description="chunk | entity | relationship | structured")
    file: str = Field(default="", description="Source filename")
    snippet: str = Field(default="", description="Relevant excerpt or description")


class AmbiguousTerm(BaseModel):
    term: str = Field(..., description="The term to be clarified")
    ambiguous: bool = Field(..., description="Whether the term is ambiguous")
    priority: str = Field(..., description="The priority of the term")


class ClarificationQuestion(BaseModel):
    term: str = Field(..., description="The term to be clarified")
    question: str = Field(
        ..., description="The question to be asked to clarify the term"
    )
    options: list[str] = Field(
        ..., description="The options to be asked to clarify the term"
    )


class ClarificationsRequest(BaseModel):
    ambiguous_terms: list[AmbiguousTerm] = Field(
        ..., description="The list of ambiguous terms"
    )
    clarification: ClarificationQuestion = Field(
        ..., description="The clarification to be asked to the user"
    )


class Clarification(BaseModel):
    term: str = Field(..., description="The term to be clarified")
    question: str = Field(
        ..., description="The question to be asked to clarify the term"
    )
    options: list[str] = Field(
        ..., description="The options to be asked to clarify the term"
    )
    user_response: str = Field(..., description="The answer to the question")


class ClarificationHistory(BaseModel):
    clarifications: list[Clarification] = Field(
        ..., description="The list of clarifications"
    )


class QueryRequest(BaseModel):
    project_id: str
    query: str
    query_type: str = "naive"
    clarification_history: list[Clarification] = Field(default_factory=list)
    number_of_clarifications: int = Field(default=0, description="The number of clarifications asked so far")
    structured_data_information: dict[str, str] = Field(default_factory=dict, description="The structured data information")
    keywords: dict[str, list[str]] = Field(default_factory=dict, description="The keywords")


class Keywords(BaseModel):
    high_level_keywords: list[str] = Field(
        default_factory=list, description="Overarching concepts or themes"
    )
    low_level_keywords: list[str] = Field(
        default_factory=list, description="Specific entities or details"
    )


class query_state(BaseModel):
    user_query: str = Field(..., description="The user's query")
    project_id: str = Field(..., description="The project ID")
    query_type: str = Field(default="naive", description="naive | local | global")
    clarifications_request: Optional[ClarificationsRequest] = Field(
        default=None, description="The clarifications request"
    )
    clarification_history: ClarificationHistory = Field(
        default_factory=lambda: ClarificationHistory(clarifications=[]),
        description="The clarification history",
    )
    number_of_clarifications: int = Field(
        default=0, description="The number of clarifications asked so far"
    )
    structured_data_information: Optional[dict[str, str]] = Field(
        default=None, description="Column name to description mapping for structured data"
    )
    file_summaries: dict[str, str] = Field(
        default_factory=dict, description="Per-file summaries generated at ingestion time"
    )
    planner_request: Optional[str] = Field(
        default=None, description="Final answer produced by the planner agent"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Sources retrieved during planning"
    )
    keywords: Optional[dict[str, list[str]]] = Field(
        default=None, description="The keywords"
    )