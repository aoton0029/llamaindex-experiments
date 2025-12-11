# src/infrastructure/storage_facade.py
"""インフラストラクチャ層の統合ファサード"""

import logging
from typing import Optional
from dataclasses import dataclass

from .vector_stores.manager import VectorStoreManager, VectorStoreConfig
from .doc_stores.manager import DocumentStoreManager, DocumentStoreConfig
from .index_stores.manager import IndexStoreManager, IndexStoreConfig
from .graph_stores.manager import GraphStoreManager, GraphStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class StorageInfraConfig:
    """インフラストラクチャ全体の設定"""
    vector_store: Optional[VectorStoreConfig] = None
    document_store: Optional[DocumentStoreConfig] = None
    index_store: Optional[IndexStoreConfig] = None
    graph_store: Optional[GraphStoreConfig] = None


class StorageFacade:
    """ストレージインフラストラクチャの統合ファサード"""
    
    def __init__(self, config: Optional[StorageInfraConfig] = None):
        self.config = config or StorageInfraConfig()
        
        # 各マネージャーの初期化
        self.vector_store_manager = VectorStoreManager(
            self.config.vector_store or VectorStoreConfig()
        )
        self.document_store_manager = DocumentStoreManager(
            self.config.document_store or DocumentStoreConfig()
        )
        self.index_store_manager = IndexStoreManager(
            self.config.index_store or IndexStoreConfig()
        )
        self.graph_store_manager = GraphStoreManager(
            self.config.graph_store or GraphStoreConfig()
        ) if self.config.graph_store else None
    
    def disconnect_all(self) -> None:
        """全ストアの接続を切断"""
        self.vector_store_manager.disconnect()
        self.document_store_manager.disconnect()
        self.index_store_manager.disconnect()
        if self.graph_store_manager:
            self.graph_store_manager.disconnect()
        logger.info("全ストレージ接続を切断しました")
    
    def health_check_all(self) -> dict:
        """全ストアのヘルスチェック"""
        return {
            "vector_store": self._check_health(self.vector_store_manager),
            "document_store": self._check_health(self.document_store_manager),
            "index_store": self._check_health(self.index_store_manager),
            "graph_store": self._check_health(self.graph_store_manager) if self.graph_store_manager else None,
        }
    
    def _check_health(self, manager) -> bool:
        try:
            manager.get_client()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False