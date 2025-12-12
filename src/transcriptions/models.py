import logging
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from pymilvus import DataType, CollectionSchema, FieldSchema

logger = logging.getLogger(__name__)


class ChunkType(str, Enum):
    """チャンクの種別"""
    SUMMARY = "summary"           # 全体概要
    TOPIC = "topic"               # トピック別要約
    CONVERSATION = "conversation"  # 会話詳細


@dataclass
class Utterance:
    """発話情報"""
    start_time: float
    end_time: float
    content: str

@dataclass
class Topic:
    """トピック情報"""
    title: str
    contents: List[str]

@dataclass
class ConversationSummary:
    """会話の概要情報"""
    summary_text: str
    topics: List[Topic]

@dataclass
class ConversationSession:
    """会話セッション"""
    uid: str
    sales_person: str
    company_name: str
    branch_name: Optional[str]
    department_name: Optional[str]
    client_person: str
    utterances: List[Utterance]
    summary: ConversationSummary
    created_at: Optional[str] = None



@dataclass
class ConversationChunkMetadata:
    """会話チャンクのメタデータ"""
    session_uid: str
    chunk_type: str  # ChunkType
    sales_person: str
    company_name: str
    branch_name: Optional[str] = None
    department_name: Optional[str] = None
    client_person: Optional[str] = None
    topic_title: Optional[str] = None  # トピックの場合のタイトル
    start_time: Optional[float] = None  # 会話チャンクの場合の開始時間
    end_time: Optional[float] = None    # 会話チャンクの場合の終了時間
    
    @staticmethod
    def schema(dim: int) -> CollectionSchema:
        """Milvus用のスキーマ定義"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="session_uid", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="sales_person", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="branch_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="department_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="client_person", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="topic_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="start_time", dtype=DataType.FLOAT),
            FieldSchema(name="end_time", dtype=DataType.FLOAT),
        ]
        return CollectionSchema(fields=fields, description="Conversation Chunk Metadata Schema")
