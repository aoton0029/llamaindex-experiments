"""
MongoDBクライアントクラス
ドキュメントストアとして使用
"""

import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.storage.docstore.types import BaseDocumentStore


logger = logging.getLogger(__name__)


class MongoDBClient:    
    def __init__(
        self, 
        database_name: str,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_source: str = "admin",
        **kwargs
    ):
        self.host = host
        self.port = port
        self.database_name = database_name
        self.username = username
        self.password = password
        self.auth_source = auth_source
        self.connection_kwargs = kwargs
        
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        self._docstore: Optional[MongoDocumentStore] = None
    
    def connect(self) -> None:
        try:
            # 接続URI構築
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}?authSource={self.auth_source}"
            else:
                uri = f"mongodb://{self.host}:{self.port}"
            
            self._client = MongoClient(uri, **self.connection_kwargs)
            self._database = self._client[self.database_name]
            
            # 接続テスト
            self._client.admin.command('ping')
            logger.info(f"MongoDB接続成功: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"MongoDB接続エラー: {e}")
            raise
    
    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            self._docstore = None
            logger.info("MongoDB接続を切断しました")
    
    def get_client(self) -> MongoClient:
        if self._client is None:
            self.connect()
        return self._client
    
    def get_database(self) -> Database:
        if self._database is None:
            self.connect()
        return self._database
    
    def get_collection(self, collection_name: str) -> Collection:
        database = self.get_database()
        return database[collection_name]
    
    def get_docstore(
        self, 
        namespace: str = "default",
        collection_name: str = "documents"
    ) -> BaseDocumentStore:
        if self._docstore is None:
            # MongoDB URIを構築
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}?authSource={self.auth_source}"
            else:
                uri = f"mongodb://{self.host}:{self.port}/{self.database_name}"
            
            self._docstore = MongoDocumentStore.from_uri(
                uri=uri,
                db_name=self.database_name,
                namespace=namespace
            )
        
        return self._docstore
    
    def create_index(
        self, 
        collection_name: str, 
        index_spec: Dict[str, Any],
        index_options: Optional[Dict[str, Any]] = None
    ) -> str:
        collection = self.get_collection(collection_name)
        return collection.create_index(index_spec, **(index_options or {}))
    
    def drop_collection(self, collection_name: str) -> None:
        database = self.get_database()
        database.drop_collection(collection_name)
        logger.info(f"コレクション '{collection_name}' を削除しました")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        database = self.get_database()
        return database.command("collStats", collection_name)
    
    def health_check(self) -> bool:
        try:
            client = self.get_client()
            client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB ヘルスチェック失敗: {e}")
            return False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def reset(self, collection_name: str) -> None:
        """指定したコレクションを削除して再作成"""
        self.drop_collection(collection_name)
        logger.info(f"コレクション '{collection_name}' をリセットしました")
        self.get_collection(collection_name)
