import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from llama_index.core.storage.storage_context import StorageContext
from .database_manager import DatabaseManager
from pymilvus import CollectionSchema
logger = logging.getLogger(__name__)

@dataclass
class DocstoreConfig:
    """ドキュメントストア設定"""
    namespace: str = "default"
    collection_name: str = "documents"


@dataclass
class IndexStoreConfig:
    """インデックスストア設定"""
    namespace: str = "default"
    collection_suffix: str = "index"


@dataclass
class VectorStoreConfig:
    """ベクトルストア設定"""
    collection_name: str
    dim: int = 8192
    schema: Optional[CollectionSchema] = None
    metric_type: str = "COSINE"
    index_type: str = "IVF_FLAT"
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class ImageStoreConfig:
    """画像ストア設定"""
    collection_name: str
    dim: int = 512
    schema: Optional[CollectionSchema] = None
    metric_type: str = "L2"
    index_type: str = "IVF_FLAT"
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class GraphStoreConfig:
    """グラフストア設定"""
    node_label: str = "Entity"
    rel_type: str = "RELATED"
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class StorageContextConfig:
    """StorageContext設定"""
    context_name: str
    docstore: Optional[DocstoreConfig] = None
    index_store: Optional[IndexStoreConfig] = None
    vector_store: Optional[VectorStoreConfig] = None
    image_store: Optional[ImageStoreConfig] = None
    graph_store: Optional[GraphStoreConfig] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StorageContextConfig':
        """
        辞書からStorageContextConfigを生成
        
        Args:
            config_dict: 設定辞書
            
        Returns:
            StorageContextConfig インスタンス
            
        Example:
            config_dict = {
                "context_name": "my_context",
                "docstore": {
                    "namespace": "docs",
                    "collection_name": "documents"
                },
                "vector_store": {
                    "collection_name": "vectors",
                    "dim": 1536
                }
            }
            config = StorageContextConfig.from_dict(config_dict)
        """
        docstore = None
        if "docstore" in config_dict and config_dict["docstore"]:
            docstore = DocstoreConfig(**config_dict["docstore"])
        
        index_store = None
        if "index_store" in config_dict and config_dict["index_store"]:
            index_store = IndexStoreConfig(**config_dict["index_store"])
        
        vector_store = None
        if "vector_store" in config_dict and config_dict["vector_store"]:
            vector_store = VectorStoreConfig(**config_dict["vector_store"])
        
        image_store = None
        if "image_store" in config_dict and config_dict["image_store"]:
            image_store = ImageStoreConfig(**config_dict["image_store"])
        
        graph_store = None
        if "graph_store" in config_dict and config_dict["graph_store"]:
            graph_store = GraphStoreConfig(**config_dict["graph_store"])
        
        return cls(
            context_name=config_dict["context_name"],
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            image_store=image_store,
            graph_store=graph_store
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        StorageContextConfigを辞書に変換
        
        Returns:
            設定辞書
        """
        result = {"context_name": self.context_name}
        
        if self.docstore:
            result["docstore"] = {
                "namespace": self.docstore.namespace,
                "collection_name": self.docstore.collection_name
            }
        
        if self.index_store:
            result["index_store"] = {
                "namespace": self.index_store.namespace,
                "collection_suffix": self.index_store.collection_suffix
            }
        
        if self.vector_store:
            result["vector_store"] = {
                "collection_name": self.vector_store.collection_name,
                "dim": self.vector_store.dim,
                "metric_type": self.vector_store.metric_type,
                "schema": self.vector_store.schema,
                "index_type": self.vector_store.index_type,
                "additional_params": self.vector_store.additional_params
            }
        
        if self.image_store:
            result["image_store"] = {
                "collection_name": self.image_store.collection_name,
                "dim": self.image_store.dim,
                "schema": self.image_store.schema,
                "metric_type": self.image_store.metric_type,
                "index_type": self.image_store.index_type,
                "additional_params": self.image_store.additional_params
            }
        
        if self.graph_store:
            result["graph_store"] = {
                "node_label": self.graph_store.node_label,
                "rel_type": self.graph_store.rel_type,
                "additional_params": self.graph_store.additional_params
            }
        
        return result

    

class StorageContextManager:
    """StorageContextの構築と管理を担当するクラス"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Args:
            db_manager: DatabaseManagerインスタンス
        """
        self.db_manager = db_manager
        self._storage_contexts: Dict[str, StorageContext] = {}
        self._storage_configs: Dict[str, StorageContextConfig] = {}
    
    def create_storage_context(self, config: StorageContextConfig) -> StorageContext:
        """
        StorageContextを作成
        """
        kwargs = {}
        if config.docstore:
            kwargs["docstore"] = self.db_manager.get_docstore(
                namespace=config.docstore.namespace,
            )
        if config.index_store:
            kwargs["index_store"] = self.db_manager.get_index_store(
                namespace=config.index_store.namespace,
                collection_suffix=config.index_store.collection_suffix
            )
        if config.vector_store:
            kwargs["vector_store"] = self.db_manager.get_vector_store(
                collection_name=config.vector_store.collection_name,
                dim=config.vector_store.dim,
                schema=config.vector_store.schema, 
            )
        if config.image_store:
            kwargs["image_store"] = self.db_manager.get_image_store(
                collection_name=config.image_store.collection_name,
                dim=config.image_store.dim,
                schema=config.image_store.schema, 
            )
        if config.graph_store:
            kwargs["graph_store"] = self.db_manager.get_graph_store(
                node_label=config.graph_store.node_label,
                rel_type=config.graph_store.rel_type
            )
        
        storage_context = StorageContext.from_defaults(**kwargs)
        
        self._storage_contexts[config.context_name] = storage_context
        self._storage_configs[config.context_name] = config
        logger.info(f"StorageContext '{config.context_name}' を作成しました")
        return storage_context
    
    def get_storage_context(self, context_name: str) -> Optional[StorageContext]:
        """作成済みのStorageContextを取得"""
        return self._storage_contexts.get(context_name)
    
    def drop_storage_context(
        self,
        context_name: str,
        vector_collection: str,
        docstore_namespace: str,
        index_namespace: str,
        image_collection: Optional[str] = None
    ) -> None:
        """
        StorageContextに関連するすべてのコレクションを削除
        
        Args:
            context_name: StorageContextの識別名
            vector_collection: ベクトルストアのコレクション名
            docstore_namespace: ドキュメントストアの名前空間
            index_namespace: インデックスストアの名前空間
            image_collection: 画像ストアのコレクション名
        """
        # MongoDB collections
        mongodb_client = self.db_manager.get_mongodb_client()
        mongodb_client.drop_collection(f"{docstore_namespace}/data")
        mongodb_client.drop_collection(f"{docstore_namespace}/metadata")
        mongodb_client.drop_collection(f"{docstore_namespace}/ref_doc_info")
        
        # Redis keys
        redis_client = self.db_manager.get_redis_client()
        redis_client.delete_key(index_namespace)
        
        # Milvus collections
        milvus_client = self.db_manager.get_milvus_client()
        milvus_client.drop_collection(vector_collection)
        if image_collection:
            milvus_client.drop_collection(image_collection)
        
        # Neo4j (必要に応じて)
        # neo4j_client = self.db_manager.get_neo4j_client()
        # neo4j_client.clear_database()
        
        # キャッシュから削除
        if context_name in self._storage_contexts:
            del self._storage_contexts[context_name]
        
        logger.info(f"StorageContext '{context_name}' を削除しました")
    
    def clear_all_storage_contexts(self) -> None:
        """すべてのStorageContextをクリア"""
        self._storage_contexts.clear()
        logger.info("すべてのStorageContextをクリアしました")
