from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pymilvus import CollectionSchema, FieldSchema, DataType
from .base import SchemaBaseModel

class GlossaryTerm(SchemaBaseModel):
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
    
    # term: str = Field(..., description="用語")
    # definition: str = Field(..., description="用語の定義")
    # category: Optional[str] = Field(None, description="用語のカテゴリ")
    # aliases: Optional[List[str]] = Field(None, description="用語の別名リスト")
    # metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ情報")

    @staticmethod
    def get_milvus_schema(dim:int) -> CollectionSchema:
        """Milvus用のコレクションスキーマを取得"""
        fields = [
            # 必須フィールド
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            
            # FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=256, description="用語"),
            # FieldSchema(name="definition", dtype=DataType.VARCHAR, max_length=1024, description="用語の定義"),
            # FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256, description="用語のカテゴリ"),
            # FieldSchema(name="aliases", dtype=DataType.VARCHAR, max_length=512, description="用語の別名リスト"),
            # FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048, description="追加のメタデータ情報"),
            # オプションフィールド
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, description="Document text content"),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256, description="Unique identifier for the document"),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256, description="Document identifier"),
            FieldSchema(name="_node_type", dtype=DataType.VARCHAR, max_length=128, description="Type of the node"),
            FieldSchema(name="_node_content", dtype=DataType.VARCHAR, max_length=65535, description="Content of the node"),
            FieldSchema(name="ref_doc_id", dtype=DataType.VARCHAR, max_length=256, description="Reference document identifier"),

            FieldSchema(name="term_name", dtype=DataType.VARCHAR, max_length=256, description="用語"),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=256, description="ファイルパス"),
            FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=256, description="タイトル"),
            FieldSchema(name="section_summary", dtype=DataType.VARCHAR, max_length=512, description="要約"),
            FieldSchema(name="excerpt_keywords", dtype=DataType.VARCHAR, max_length=512, description="キーワード"),
        ]
        schema = CollectionSchema(fields=fields, description="Glossary Terms Collection")
        return schema
