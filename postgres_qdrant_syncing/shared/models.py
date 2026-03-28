from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel


class ProductBase(BaseModel):
    article_id: str
    name: str
    description: Optional[str] = None
    product_type: Optional[str] = None
    product_group: Optional[str] = None
    color: Optional[str] = None
    department: Optional[str] = None
    index_name: Optional[str] = None
    image_url: Optional[str] = None
    price: Optional[Decimal] = None


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    """
    Full replacement — all fields required except article_id (comes from URL).
    """
    name: str
    description: Optional[str] = None
    product_type: Optional[str] = None
    product_group: Optional[str] = None
    color: Optional[str] = None
    department: Optional[str] = None
    index_name: Optional[str] = None
    image_url: Optional[str] = None
    price: Optional[Decimal] = None


class ProductPatch(BaseModel):
    """
    Partial update — all fields optional.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    product_type: Optional[str] = None
    product_group: Optional[str] = None
    color: Optional[str] = None
    department: Optional[str] = None
    index_name: Optional[str] = None
    image_url: Optional[str] = None
    price: Optional[Decimal] = None


class ProductResponse(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SearchResult(BaseModel):
    article_id: str
    name: str
    description: Optional[str] = None
    product_type: Optional[str] = None
    color: Optional[str] = None
    department: Optional[str] = None
    price: Optional[Decimal] = None
    score: float
    payload: dict[str, Any] = {}


class ReconcileResult(BaseModel):
    postgres_count: int
    qdrant_count: int
    missing_in_qdrant: int
    orphaned_in_qdrant: int
    in_sync: bool
    missing_ids: list[str] = []
    orphaned_ids: list[str] = []


class SyncStatus(BaseModel):
    pending_events: int
    failed_events: int
    completed_last_hour: int
    avg_lag_seconds: Optional[float] = None