from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pymilvus import CollectionSchema, FieldSchema, DataType
from .base import SchemaBaseModel

class TechColumnTerm(SchemaBaseModel):
    """技術コラム"""
    
    @staticmethod
    def get_milvus_schema(dim:int) -> CollectionSchema:
        """Milvus用のコレクションスキーマを取得"""
        fields = [
            # 必須フィールド
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            
            # オプションフィールド
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, description="Document text content"),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256, description="Unique identifier for the document"),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256, description="Document identifier"),
            FieldSchema(name="_node_type", dtype=DataType.VARCHAR, max_length=128, description="Type of the node"),
            FieldSchema(name="_node_content", dtype=DataType.VARCHAR, max_length=65535, description="Content of the node"),
            FieldSchema(name="ref_doc_id", dtype=DataType.VARCHAR, max_length=256, description="Reference document identifier"),
            # エクストラクタ
            # FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=512, description="タイトル"),
            # FieldSchema(name="section_summary", dtype=DataType.VARCHAR, max_length=1024, description="要約"),
            # FieldSchema(name="excerpt_keywords", dtype=DataType.VARCHAR, max_length=1024, description="キーワード"),
            # メタデータ
            FieldSchema(name="term_name", dtype=DataType.VARCHAR, max_length=256, description="用語"),
        ]
        schema = CollectionSchema(fields=fields, description="Tech Column Collection")
        return schema