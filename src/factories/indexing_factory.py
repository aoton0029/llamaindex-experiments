
import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from llama_index.core.schema import BaseNode, Document
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    DocumentSummaryIndex
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.base import BaseIndex

logger = logging.getLogger(__name__)


class IndexBuilder(ABC):
    """
    インデックスビルダー基底クラス
    """
    
    def __init__(
        self,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = True,
        **kwargs
    ):
        self.storage_context = storage_context
        self.show_progress = show_progress
        self.kwargs = kwargs
        self._index = None
    
    @abstractmethod
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        pass
    
    @abstractmethod
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        pass
    
    def get_index(self) -> Optional[BaseIndex]:
        """構築されたインデックスを取得"""
        return self._index