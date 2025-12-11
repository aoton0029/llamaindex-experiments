# src/infrastructure/vector_stores/manager.py
"""ベクトルストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.vector_stores.types import VectorStore
from .milvus_client import MilvusClient

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """ベクトルストア設定"""
    host: str = "milvus"
    port: int = 19530
    user: Optional[str] = "admin"
    password: Optional[str] = "pdntsPa0"


class VectorStoreManager:
    """ベクトルストアの管理を担当"""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._client: Optional[MilvusClient] = None
    
    def get_client(self) -> MilvusClient:
        """Milvusクライアントを取得"""
        if self._client is None:
            self._client = MilvusClient(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password
            )
        return self._client
    
    def get_vector_store(
        self,
        collection_name: str,
        dim: int,
        **kwargs
    ) -> VectorStore:
        """ベクトルストアを取得"""
        client = self.get_client()
        return client.get_vector_store(
            collection_name=collection_name,
            dim=dim,
            **kwargs
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None