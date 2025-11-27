"""
データベース接続管理クラス
各種データベースクライアントを統合管理
"""

import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from pymilvus import CollectionSchema

from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.graph_stores.types import GraphStore

from .mongodb_client import MongoDBClient
from .redis_client import RedisClient
from .milvus_client import MilvusClient
from .neo4j_client import Neo4jClient


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """データベース設定クラス"""
    username: str = "admin"
    password: str = "pdntsPa0"
    
    # MongoDB設定
    mongodb_base_url: str = "mongodb://admin:pdntsPa0@mongodb:27017"
    mongodb_host: str = "mongodb"
    mongodb_port: int = 27017
    mongodb_database: str = "pdf_rag_system"
    mongodb_username: Optional[str] = username
    mongodb_password: Optional[str] = password
    
    # Redis設定
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: Optional[str] = password
    redis_db: int = 0
    
    # Milvus設定
    milvus_host: str = "milvus"
    milvus_port: int = 19530
    milvus_user: Optional[str] = username
    milvus_password: Optional[str] = password
    
    # Neo4j設定
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = password
    neo4j_database: str = "neo4j"


class DatabaseManager:    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        
        # クライアントインスタンス
        self._mongodb_client: Optional[MongoDBClient] = None
        self._redis_client: Optional[RedisClient] = None
        self._milvus_client: Optional[MilvusClient] = None
        self._neo4j_client: Optional[Neo4jClient] = None
    
    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> 'DatabaseManager':
        config = DatabaseConfig(**config_dict)
        return cls(config)
    

    
    def get_mongodb_client(self) -> MongoDBClient:
        """MongoDBクライアントを取得"""
        if self._mongodb_client is None:
            self._mongodb_client = MongoDBClient(
                host=self.config.mongodb_host,
                port=self.config.mongodb_port,
                database_name=self.config.mongodb_database,
                username=self.config.mongodb_username,
                password=self.config.mongodb_password
            )
        return self._mongodb_client
    
    def get_redis_client(self) -> RedisClient:
        """Redisクライアントを取得"""
        if self._redis_client is None:
            self._redis_client = RedisClient(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db
            )
        return self._redis_client
    
    def get_milvus_client(self) -> MilvusClient:
        """Milvusクライアントを取得"""
        if self._milvus_client is None:
            self._milvus_client = MilvusClient(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                user=self.config.milvus_user,
                password=self.config.milvus_password
            )
        return self._milvus_client
    
    def get_neo4j_client(self) -> Neo4jClient:
        """Neo4jクライアントを取得"""
        if self._neo4j_client is None:
            self._neo4j_client = Neo4jClient(
                uri=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
        return self._neo4j_client
    
    def get_docstore(
        self, 
        namespace: str = "default",
        collection_name: str = "documents"
    ) -> BaseDocumentStore:
        """MongoDBベースのドキュメントストアを取得（毎回新規作成）"""
        mongodb_client = self.get_mongodb_client()
        return mongodb_client.get_docstore(
            namespace=namespace,
            collection_name=collection_name
        )
    
    def get_index_store(
        self,
        namespace: str,
        collection_suffix: str
    ) -> BaseIndexStore:
        """Redisベースのインデックスストアを取得（毎回新規作成）"""
        redis_client = self.get_redis_client()
        return redis_client.get_index_store(
            namespace=namespace,
            collection_suffix=collection_suffix
        )
    
    def get_vector_store(
        self,
        collection_name: str,
        dim: int,
        **kwargs
    ) -> VectorStore:
        """Milvusベースのベクトルストアを取得（毎回新規作成）"""
        milvus_client = self.get_milvus_client()
        return milvus_client.get_vector_store(
            collection_name=collection_name,
            dim=dim,
            **kwargs
        )

    def get_image_store(
        self,
        collection_name: str,
        dim: int,
        **kwargs
    ) -> Optional[VectorStore]:
        """Milvusベースの画像ストアを取得（毎回新規作成）"""
        if collection_name is None:
            return None
        milvus_client = self.get_milvus_client()
        return milvus_client.get_vector_store(
            collection_name=collection_name,
            dim=dim,
            **kwargs
        )
    
    def get_graph_store(
        self,
        node_label: str = "Entity",
        rel_type: str = "RELATED",
        **kwargs
    ) -> GraphStore:
        """Neo4jベースのグラフストアを取得（毎回新規作成）"""
        neo4j_client = self.get_neo4j_client()
        return neo4j_client.get_graph_store(
            node_label=node_label,
            rel_type=rel_type,
            **kwargs
        )
        
    def connect_all(self) -> None:
        """全データベースに接続"""
        try:
            self.get_mongodb_client().connect()
            logger.info("MongoDB接続完了")
        except Exception as e:
            logger.error(f"MongoDB接続失敗: {e}")
        
        try:
            self.get_redis_client().connect()
            logger.info("Redis接続完了")
        except Exception as e:
            logger.error(f"Redis接続失敗: {e}")
        
        try:
            self.get_milvus_client().connect()
            logger.info("Milvus接続完了")
        except Exception as e:
            logger.error(f"Milvus接続失敗: {e}")
        
        try:
            self.get_neo4j_client().connect()
            logger.info("Neo4j接続完了")
        except Exception as e:
            logger.error(f"Neo4j接続失敗: {e}")
    
    def disconnect_all(self) -> None:
        """全データベースから切断"""
        if self._mongodb_client:
            self._mongodb_client.disconnect()
        if self._redis_client:
            self._redis_client.disconnect()
        if self._milvus_client:
            self._milvus_client.disconnect()
        if self._neo4j_client:
            self._neo4j_client.disconnect()
        
        logger.info("全データベース接続を切断しました")
    
    def health_check_all(self) -> Dict[str, bool]:
        """全データベースのヘルスチェック"""
        health_status = {}
        
        try:
            health_status["mongodb"] = self.get_mongodb_client().health_check()
        except Exception as e:
            logger.error(f"MongoDB ヘルスチェックエラー: {e}")
            health_status["mongodb"] = False
        
        try:
            health_status["redis"] = self.get_redis_client().health_check()
        except Exception as e:
            logger.error(f"Redis ヘルスチェックエラー: {e}")
            health_status["redis"] = False
        
        try:
            health_status["milvus"] = self.get_milvus_client().health_check()
        except Exception as e:
            logger.error(f"Milvus ヘルスチェックエラー: {e}")
            health_status["milvus"] = False
        
        try:
            health_status["neo4j"] = self.get_neo4j_client().health_check()
        except Exception as e:
            logger.error(f"Neo4j ヘルスチェックエラー: {e}")
            health_status["neo4j"] = False
        
        return health_status

    
    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.connect_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.disconnect_all()

