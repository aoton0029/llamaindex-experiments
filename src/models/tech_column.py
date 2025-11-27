from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pymilvus import CollectionSchema, FieldSchema, DataType
from .base import SchemaBaseModel

class TechColumnTerm(SchemaBaseModel):
    """
    Tech column model

    "title": "Understanding Neural Networks",
    "author": "John Doe",
    "content": "ニューラルネットワークは...",
    "tags": ["AI", "Machine Learning"],
    "published_date": "2025-10-15",
    "metadata": {
      "source": "Tech Blog",
      "last_updated": "2025-11-01"
    }
    """
    # title: str = Field(..., description="コラムのタイトル")
    # author: str = Field(..., description="著者名")
    # content: str = Field(..., description="コラムの内容")
    # tags: List[str] = Field(..., description="関連タグ")
    # published_date: str = Field(..., description="公開日")
    # metadata: Dict[str, Any] = Field(..., description="メタデータ")

    @staticmethod
    def get_milvus_schema(dim:int) -> CollectionSchema:
        """Milvus用のコレクションスキーマを取得"""
        fields = [
            # 必須フィールド
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            # オプションフィールド
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256, description="コラムのタイトル"),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=128, description="著者名"),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096, description="コラムの内容"),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=512, description="関連タグ"),
            FieldSchema(name="published_date", dtype=DataType.VARCHAR, max_length=32, description="公開日"),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048, description="メタデータ"),
        ]
        schema = CollectionSchema(fields=fields, description="Tech Column Collection")
        return schema
