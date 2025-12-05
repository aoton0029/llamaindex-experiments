"""
Milvusクライアントクラス
ベクトルストアとして使用
"""

import logging
from typing import Optional, Dict, Any, List
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores.types import VectorStore


logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connection_kwargs = kwargs
        
        self._connection_alias = f"milvus_{self.host}_{self.port}"
        self._collections: Dict[str, Collection] = {}
        self._vector_stores: Dict[str, MilvusVectorStore] = {}
    
    def connect(self) -> None:
        try:
            # 既に接続済みの場合はスキップ
            if connections.has_connection(self._connection_alias):
                logger.info(f"Milvus接続済み: {self.host}:{self.port}")
                return
            
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
            if connections.has_connection(self._connection_alias):
                connections.disconnect(alias=self._connection_alias)
            self._collections.clear()
            self._vector_stores.clear()
            logger.info("Milvus接続を切断しました")
        except Exception as e:
            logger.warning(f"Milvus切断時警告: {e}")
    
    def create_collection(
        self, 
        collection_name: str,
        dim: int,
        schema: Optional[CollectionSchema],
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        nlist: int = 1024
    ) -> Collection:
        if not connections.has_connection(self._connection_alias):
            self.connect()
        
        # 既存コレクションをチェック
        if utility.has_collection(collection_name, using=self._connection_alias):
            logger.info(f"コレクション '{collection_name}' は既に存在します")
            collection = Collection(collection_name, using=self._connection_alias)
            self._collections[collection_name] = collection
            return collection
        
        # コレクション作成
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self._connection_alias
        )
        
        # インデックス作成
        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {"nlist": nlist}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        self._collections[collection_name] = collection
        logger.info(f"コレクション '{collection_name}' を作成しました (dim={dim})")
        return collection
    
    def get_collection(self, collection_name: str) -> Collection:
        if collection_name not in self._collections:
            if not connections.has_connection(self._connection_alias):
                self.connect()
            self._collections[collection_name] = Collection(collection_name, using=self._connection_alias)
        return self._collections[collection_name]
    
    def get_vector_store(
        self,
        collection_name: str,
        dim: int,
        schema: Optional[CollectionSchema] = None,
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        nlist: int = 1024,
        **kwargs
    ) -> VectorStore:
        """
        VectorStoreを取得 (コレクションごとにキャッシュ)
        
        Args:
            collection_name: コレクション名
            dim: ベクトル次元数
            index_type: インデックスタイプ
            metric_type: 距離メトリック
            nlist: IVF_FLATのクラスター数
            **kwargs: 追加パラメータ
            
        Returns:
            VectorStore
        """
        if collection_name not in self._vector_stores:
            if not connections.has_connection(self._connection_alias):
                self.connect()
            
            # コレクションが存在しない場合は作成
            if not utility.has_collection(collection_name, using=self._connection_alias):
                self.create_collection(
                    collection_name=collection_name,
                    dim=dim,
                    schema=schema,
                    index_type=index_type,
                    metric_type=metric_type,
                    nlist=nlist
                )
            
            self._vector_stores[collection_name] = MilvusVectorStore(
                uri=f"http://{self.host}:{self.port}",
                token=f"{self.user}:{self.password}" if self.user and self.password else None,
                collection_name=collection_name,
                dim=dim,
                **kwargs
            )
        
        return self._vector_stores[collection_name]
    
    def load_collection(self, collection_name: str) -> None:
        """コレクションをメモリにロード"""
        collection = self.get_collection(collection_name)
        collection.load()
        logger.info(f"コレクション '{collection.name}' をロードしました")
    
    def release_collection(self, collection_name: str) -> None:
        """コレクションをメモリから解放"""
        collection = self.get_collection(collection_name)
        collection.release()
        logger.info(f"コレクション '{collection.name}' を解放しました")
    
    def drop_collection(self, collection_name: str) -> None:
        """コレクションを削除"""
        if utility.has_collection(collection_name, using=self._connection_alias):
            utility.drop_collection(collection_name, using=self._connection_alias)
            if collection_name in self._collections:
                del self._collections[collection_name]
            if collection_name in self._vector_stores:
                del self._vector_stores[collection_name]
            logger.info(f"コレクション '{collection_name}' を削除しました")
    
    def list_collections(self) -> List[str]:
        """全コレクション名を取得"""
        if not connections.has_connection(self._connection_alias):
            self.connect()
        return utility.list_collections(using=self._connection_alias)
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """コレクション統計を取得"""
        collection = self.get_collection(collection_name)
        stats = {
            "num_entities": collection.num_entities,
            "schema": collection.schema,
            "indexes": collection.indexes
        }
        return stats
    
    def health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            if not connections.has_connection(self._connection_alias):
                self.connect()
            return True
        except Exception as e:
            logger.error(f"Milvus ヘルスチェック失敗: {e}")
            return False
    
    def reset(self, collection_name: Optional[str] = None) -> None:
        """
        コレクションをリセット
        
        Args:
            collection_name: コレクション名 (Noneの場合は全コレクション)
        """
        if collection_name:
            self.drop_collection(collection_name)
        else:
            # 全コレクションを削除
            for name in self.list_collections():
                self.drop_collection(name)
            logger.info("全コレクションをリセットしました")
    
    
    def get_field_values(self, collection_name, expr, fields:List[str], limit:int):
        collection = self.get_collection(collection_name)        
        results = collection.query(expr=expr, output_fields=fields, limit=limit)
        logger.info(f"{collection_name}から{len(results)}件取得")
        return results
                
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
