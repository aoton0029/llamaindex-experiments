from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pymilvus import CollectionSchema, FieldSchema, DataType
from .base import SchemaBaseModel

class PdfDocumentVector(SchemaBaseModel):
    """
    PDFドキュメントベクトルモデル    
    """
    # _id : Optional[int] = Field(None, description="Primary key ID")
    # id: str = Field(..., description="Document vector ID")
    # embedding: List[float] = Field(..., description="Document vector embedding") 
    # text: str = Field(..., description="Document text content")
    # document_id: str = Field(..., description="Unique identifier for the document")
    # doc_id: str = Field(..., description="Document identifier")
    # _node_type: str = Field(..., description="Type of the node")
    # _node_content: str = Field(..., description="Content of the node")
    # ref_doc_id: str = Field(..., description="Reference document identifier")    
    # # メタデータ
    # title: Optional[str] = Field(None, description="Title of the document")
    # page: Optional[int] = Field(None, description="Page number of the document")
    # total_pages: Optional[int] = Field(None, description="Total number of pages in the document")
    # file_path: Optional[str] = Field(None, description="File path of the document")

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
            # メタデータ
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024, description="Title of the document"),
            FieldSchema(name="page", dtype=DataType.INT64, description="Page number of the document"),
            FieldSchema(name="total_pages", dtype=DataType.INT64, description="Total number of pages in the document"),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=2048, description="File path of the document"),
        ]
        schema = CollectionSchema(fields=fields, description="PDF Document Vector Collection")
        return schema

