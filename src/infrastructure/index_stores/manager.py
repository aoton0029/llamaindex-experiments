# src/infrastructure/index_stores/manager.py
"""インデックスストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.storage.index_store.types import BaseIndexStore
from .redis_client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class IndexStoreConfig:
    """インデックスストア設定"""
    host: str = "redis-store"
    port: int = 6379
    password: Optional[str] = "pdntsPa0"
    db: int = 0


class IndexStoreManager:
    """インデックスストアの管理を担当"""
    
    def __init__(self, config: Optional[IndexStoreConfig] = None):
        self.config = config or IndexStoreConfig()
        self._client: Optional[RedisClient] = None
    
    def get_client(self) -> RedisClient:
        """Redisクライアントを取得"""
        if self._client is None:
            self._client = RedisClient(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db
            )
        return self._client
    
    def get_index_store(
        self,
        namespace: str = "default",
        collection_suffix: str = ""
    ) -> BaseIndexStore:
        """インデックスストアを取得"""
        client = self.get_client()
        return client.get_index_store(
            namespace=namespace,
            collection_suffix=collection_suffix
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None