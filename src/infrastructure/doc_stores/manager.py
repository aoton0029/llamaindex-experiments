# src/infrastructure/document_stores/manager.py
"""ドキュメントストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.storage.docstore.types import BaseDocumentStore
from .mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


@dataclass
class DocumentStoreConfig:
    """ドキュメントストア設定"""
    host: str = "mongodb"
    port: int = 27017
    database_name: str = "pdf_rag_system"
    username: Optional[str] = "admin"
    password: Optional[str] = "pdntsPa0"
    auth_source: str = "admin"


class DocumentStoreManager:
    """ドキュメントストアの管理を担当"""
    
    def __init__(self, config: Optional[DocumentStoreConfig] = None):
        self.config = config or DocumentStoreConfig()
        self._client: Optional[MongoDBClient] = None
    
    def get_client(self) -> MongoDBClient:
        """MongoDBクライアントを取得"""
        if self._client is None:
            self._client = MongoDBClient(
                host=self.config.host,
                port=self.config.port,
                database_name=self.config.database_name,
                username=self.config.username,
                password=self.config.password,
                auth_source=self.config.auth_source
            )
        return self._client
    
    def get_docstore(
        self,
        namespace: str = "default",
        collection_name: str = "documents"
    ) -> BaseDocumentStore:
        """ドキュメントストアを取得"""
        client = self.get_client()
        return client.get_docstore(
            namespace=namespace,
            collection_name=collection_name
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None