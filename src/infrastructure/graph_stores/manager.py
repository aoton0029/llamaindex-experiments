# src/infrastructure/graph_stores/manager.py
"""グラフストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.graph_stores.types import GraphStore
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class GraphStoreConfig:
    """グラフストア設定"""
    uri: str = "bolt://neo4j:7687"
    username: str = "neo4j"
    password: str = "pdntsPa0"
    database: str = "neo4j"


class GraphStoreManager:
    """グラフストアの管理を担当"""
    
    def __init__(self, config: Optional[GraphStoreConfig] = None):
        self.config = config or GraphStoreConfig()
        self._client: Optional[Neo4jClient] = None
    
    def get_client(self) -> Neo4jClient:
        """Neo4jクライアントを取得"""
        if self._client is None:
            self._client = Neo4jClient(
                uri=self.config.uri,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database
            )
        return self._client
    
    def get_graph_store(
        self,
        node_label: str = "Entity",
        rel_type: str = "RELATED",
        **kwargs
    ) -> GraphStore:
        """グラフストアを取得"""
        client = self.get_client()
        return client.get_graph_store(
            node_label=node_label,
            rel_type=rel_type,
            **kwargs
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None