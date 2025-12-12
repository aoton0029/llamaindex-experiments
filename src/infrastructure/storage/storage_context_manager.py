import logging
from typing import Optional, Dict, List
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage, load_indices_from_storage

from .vector_stores.manager import VectorStoreManager
from .infrastructure.doc_stores.manager import DocumentStoreManager
from .infrastructure.graph_stores.manager import GraphStoreManager
from .infrastructure.index_stores.manager import IndexStoreManager
from .storage_context_config import StorageContextConfig


logger = logging.getLogger(__name__)


class StorageContextManager:
    """StorageContextの構築と管理を担当するクラス"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        document_store_manager: DocumentStoreManager,
        index_store_manager: IndexStoreManager,
        graph_store_manager: Optional[GraphStoreManager] = None
    ):
        self.vector_store_manager = vector_store_manager
        self.document_store_manager = document_store_manager
        self.index_store_manager = index_store_manager
        self.graph_store_manager = graph_store_manager
        
        self._storage_contexts: Dict[str, StorageContext] = {}
        self._storage_configs: Dict[str, StorageContextConfig] = {}
    
    def create_storage_context(self, config: StorageContextConfig) -> StorageContext:
        """StorageContextを作成"""
        kwargs = {}
        
        if config.docstore:
            kwargs["docstore"] = self.document_store_manager.get_docstore(
                namespace=config.docstore.namespace,
            )
        
        if config.index_store:
            kwargs["index_store"] = self.index_store_manager.get_index_store(
                namespace=config.index_store.namespace,
                collection_suffix=config.index_store.collection_suffix
            )
        
        if config.vector_store:
            kwargs["vector_store"] = self.vector_store_manager.get_vector_store(
                collection_name=config.vector_store.collection_name,
                dim=config.vector_store.dim,
                schema=config.vector_store.schema,
            )
            try:
                milvus_client = self.vector_store_manager.get_client()
                milvus_client.load_collection(config.vector_store.collection_name)
                logger.info(f"Successfully loaded Milvus Collection: {config.vector_store.collection_name}")
            except Exception as e:
                logger.info(f"collection loaded Error: {config.vector_store.collection_name}")
        
        if config.image_store:
            kwargs["image_store"] = self.vector_store_manager.get_vector_store(
                collection_name=config.image_store.collection_name,
                dim=config.image_store.dim,
                schema=config.image_store.schema,
            )
        
        if config.graph_store and self.graph_store_manager:
            kwargs["graph_store"] = self.graph_store_manager.get_graph_store(
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
    
    def load_index(self, context_name: str, index_id: Optional[str]) -> BaseIndex | List[BaseIndex]:
        """StorageContextからインデックスをロード"""
        storage_context = self.get_storage_context(context_name)
        if not storage_context:
            raise ValueError(f"StorageContext '{context_name}' が見つかりません")
        
        index = load_index_from_storage(storage_context, index_id=index_id)
        logger.info(f"インデックス '{index_id}' をロードしました")
        return index
    
    def load_indices(self, context_name: str) -> List[BaseIndex]:
        """全インデックスをロード"""
        storage_context = self.get_storage_context(context_name)
        if not storage_context:
            raise ValueError(f"StorageContext '{context_name}' が見つかりません")
        
        indices = load_indices_from_storage(storage_context)
        logger.info(f"{len(indices)} 個のインデックスをロードしました")
        return indices

    def load_indices_to_dict(self, context_name:str) -> Dict[str, BaseIndex]:
        """インデックスのタイプごとに辞書で取得"""
        storage_context = self.get_storage_context(context_name)
        if not storage_context:
            raise ValueError(f"StorageContext '{context_name}' が見つかりません")

        indices = load_indices_from_storage(storage_context)
        index_dict = {}
        for index in indices:
            index_type = type(index).__name__
            index_dict[index_type] = index
        return index_dict
        
    
    def drop_storage_context_by_name(self, context_name: str) -> None:
        """
        StorageContextに関連するすべてのコレクションを削除
        
        Args:
            context_name: StorageContextの識別名
        """
        config = self._storage_configs.get(context_name)
        if not config:
            logger.warning(f"StorageContextConfigが見つかりません: {context_name}")
            return
        self.drop_storage_context(config)

    def drop_storage_context(self, config: StorageContextConfig) -> None:
        """
        StorageContextに関連するすべてのコレクションを削除

        Args:
            config: StorageContextConfigインスタンス
        """
        context_name = config.context_name

        # MongoDB collections
        if self.document_store_manager:
            client = self.document_store_manager.get_client()
            client.drop_collection(f"{config.docstore.namespace}/data")
            client.drop_collection(f"{config.docstore.namespace}/metadata")
            client.drop_collection(f"{config.docstore.namespace}/ref_doc_info")

        # Redis keys
        if self.index_store_manager:
            client = self.index_store_manager.get_client()
            client.delete_key(f"{config.index_store.namespace}/data")

        # Milvus collections
        if self.vector_store_manager:
            milvus_client = self.vector_store_manager.get_client()
            milvus_client.drop_collection(f"{config.vector_store.collection_name}")
            if config.image_store:
                milvus_client.drop_collection(f"{config.image_store.collection_name}")

        # Neo4j
        # if self.graph_store_manager:
        #     graph_client = self.graph_store_manager.get_client()
        #     graph_client.clear_graph()
        
        # キャッシュから削除
        if context_name in self._storage_contexts:
            del self._storage_contexts[context_name]
        
        logger.info(f"StorageContext '{context_name}' を削除しました")
    
    def clear_all_storage_contexts(self) -> None:
        """すべてのStorageContextをクリア"""
        self._storage_contexts.clear()
        logger.info("すべてのStorageContextをクリアしました")