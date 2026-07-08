from __future__ import annotations

from pydantic import BaseModel, Field


class CorrelationRequest(BaseModel):
    dataset_id: str = Field(description="数据集 ID")
    col1: str = Field(description="First column name")
    col2: str = Field(description="Second column name")


class ErrorPayload(BaseModel):
    code: str
    message: str
