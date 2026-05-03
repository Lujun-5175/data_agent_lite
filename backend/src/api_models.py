from __future__ import annotations

from pydantic import BaseModel, Field


class CorrelationRequest(BaseModel):
    dataset_id: str = Field(description="数据集 ID")
    col1: str = Field(description="第一列名称")
    col2: str = Field(description="第二列名称")


class ErrorPayload(BaseModel):
    code: str
    message: str
