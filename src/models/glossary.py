from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pymilvus import CollectionSchema, FieldSchema, DataType

class GlossaryTerm(BaseModel):
    """
    Glossary term model
    
    "term": "Machine Learning",
    "definition": "コンピュータが経験から自動的に学習する技術",
    "category": "AI",
    "aliases": ["機械学習", "ML"],
    "metadata": {
      "source": "AI用語集",
      "last_updated": "2025-11-22"
    }
    """
    term: str = Field(..., description="用語")
    definition: str = Field(..., description="用語の定義")
    category: Optional[str] = Field(None, description="用語のカテゴリ")
    aliases: Optional[List[str]] = Field(None, description="用語の別名リスト")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ情報")

    @staticmethod
    def get_milvus_schema() -> CollectionSchema:
        """Milvus用のコレクションスキーマを取得"""
        fields = [
            FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=256, is_primary=True, description="用語"),
            FieldSchema(name="definition", dtype=DataType.VARCHAR, max_length=1024, description="用語の定義"),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256, description="用語のカテゴリ"),
            FieldSchema(name="aliases", dtype=DataType.VARCHAR, max_length=512, description="用語の別名リスト"),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048, description="追加のメタデータ情報"),
        ]
        schema = CollectionSchema(fields=fields, description="Glossary Terms Collection")
        return schema