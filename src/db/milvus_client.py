"""
Milvusクライアントクラス
ベクトルストアとして使用
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.types import VectorStore


logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(
        self,
        collection_name: str,
        dim: int,
        collection_schema: Optional[CollectionSchema] = None,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        nlist: int = 1024,
        **kwargs
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.dim = dim
        self.collection_schema = collection_schema
        self.index_type = index_type
        self.metric_type = metric_type
        self.nlist = nlist
        self.connection_kwargs = kwargs
        
        self._connection_alias = f"milvus_{self.host}_{self.port}"
        self._collection: Optional[Collection] = None
        self._vector_store: Optional[MilvusVectorStore] = None
    
    def connect(self) -> None:
        try:
            # 接続パラメータ構築
            connect_params = {
                "alias": self._connection_alias,
                "host": self.host,
                "port": self.port,
                **self.connection_kwargs
            }
            
            if self.user and self.password:
                connect_params["user"] = self.user
                connect_params["password"] = self.password
            
            # Milvus接続
            connections.connect(**connect_params)
            
            logger.info(f"Milvus接続成功: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Milvus接続エラー: {e}")
            raise
    
    def disconnect(self) -> None:
        try:
            connections.disconnect(alias=self._connection_alias)
            self._collection = None
            self._vector_store = None
            logger.info("Milvus接続を切断しました")
        except Exception as e:
            logger.warning(f"Milvus切断時警告: {e}")
    
    def create_collection(
        self, 
        collection_name: Optional[str] = None,
        schema: Optional[CollectionSchema] = None
    ) -> Collection:
        if not connections.has_connection(self._connection_alias):
            self.connect()
        
        name = collection_name or self.collection_name
        
        # 既存コレクションをチェック
        if utility.has_collection(name, using=self._connection_alias):
            logger.info(f"コレクション '{name}' は既に存在します")
            return Collection(name, using=self._connection_alias)
        
        # スキーマが指定されていない場合はデフォルトスキーマを作成
        if schema is None:
            schema = self.collection_schema
        
        # コレクション作成
        collection = Collection(
            name=name,
            schema=schema,
            using=self._connection_alias
        )
        
        logger.info(f"コレクション '{name}' を作成しました")
        return collection
    
    def get_collection(self, collection_name: Optional[str] = None) -> Collection:
        if self._collection is None:
            name = collection_name or self.collection_name
            self._collection = Collection(name, using=self._connection_alias)
        return self._collection
    
    def create_index(
        self, 
        collection: Optional[Collection] = None,
        field_name: str = "vector"
    ) -> None:
        if collection is None:
            collection = self.get_collection()
        
        index_params = {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": {"nlist": self.nlist}
        }
        
        collection.create_index(
            field_name=field_name,
            index_params=index_params
        )
        
        logger.info(f"インデックスを作成しました: {field_name}")
    
    def get_vector_store(
        self,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> VectorStore:
        if self._vector_store is None:
            if not connections.has_connection(self._connection_alias):
                self.connect()
            
            name = collection_name or self.collection_name
            
            # コレクションが存在しない場合は作成
            if not utility.has_collection(name, using=self._connection_alias):
                collection = self.create_collection(name)
                self.create_index(collection)
            
            self._vector_store = MilvusVectorStore(
                uri=f"http://{self.host}:{self.port}",
                token=f"{self.user}:{self.password}" if self.user and self.password else None,
                collection_name=name,
                dim=self.dim,
                **kwargs
            )
        
        return self._vector_store
    
    def load_collection(self, collection_name: Optional[str] = None) -> None:
        collection = self.get_collection(collection_name)
        collection.load()
        logger.info(f"コレクション '{collection.name}' をロードしました")
    
    def release_collection(self, collection_name: Optional[str] = None) -> None:
        collection = self.get_collection(collection_name)
        collection.release()
        logger.info(f"コレクション '{collection.name}' を解放しました")
    
    def drop_collection(self, collection_name: Optional[str] = None) -> None:
        name = collection_name or self.collection_name
        if utility.has_collection(name, using=self._connection_alias):
            utility.drop_collection(name, using=self._connection_alias)
            logger.info(f"コレクション '{name}' を削除しました")
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        collection = self.get_collection(collection_name)
        stats = {
            "num_entities": collection.num_entities,
            "schema": collection.schema,
            "indexes": collection.indexes
        }
        return stats
    
    def health_check(self) -> bool:
        try:
            if not connections.has_connection(self._connection_alias):
                self.connect()
            return True
        except Exception as e:
            logger.error(f"Milvus ヘルスチェック失敗: {e}")
            return False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def reset(self, collection_name: Optional[str] = None):
        name = collection_name or self.collection_name
        self.drop_collection(name)
        self.create_collection(name)