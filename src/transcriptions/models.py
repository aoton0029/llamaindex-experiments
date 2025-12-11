from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from pymilvus import CollectionSchema, FieldSchema, DataType

@dataclass
class Utterance:
    """発話情報"""
    start_time: float  # 発話開始時間（秒）
    end_time: float    # 発話終了時間（秒）
    content: str       # 発話内容

@dataclass
class ConversationSession:
    """会話セッション"""
    uid: str                              # 会話セッションのユニークID
    salesperson_name: Optional[str]       # 営業担当者名
    company_name: Optional[str]           # 会社名
    branch_name: Optional[str]            # 拠点名
    department_name: Optional[str]        # 部署名
    client_contact_name: Optional[str]    # 取引先担当者名
    utterances: List[Utterance]           # 発話情報のリスト
    summary: Optional[Dict[str, str]]     # 要約（全体概要、トピック別要約、決定事項）


class ConversationChunkMetadata:
    """会話チャンクのメタデータスキーマ定義（Milvus用）"""
    
    @staticmethod
    def schema(dim: int) -> CollectionSchema:
        """
        Milvus用のスキーマ定義
        
        Args:
            dim: Embeddingベクトルの次元数
        """        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="uid", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="salesperson_name", dtype=DataType.VARCHAR, max_length=128),
        ]
        return CollectionSchema(fields=fields, description="Conversation Chunk Metadata Schema")
