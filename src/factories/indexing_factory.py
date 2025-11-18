
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
    DocumentSummaryIndex,
)
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.base import BaseIndex

logger = logging.getLogger(__name__)


class IndexBuilder(ABC):
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

class VectorStoreIndexBuilder(IndexBuilder):
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index


class SummaryIndexBuilder(IndexBuilder):
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = SummaryIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = SummaryIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index

class TreeIndexBuilder(IndexBuilder):
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = TreeIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = TreeIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index

class KeywordTableIndexBuilder(IndexBuilder):
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = KeywordTableIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = KeywordTableIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index

class KnowledgeGraphIndexBuilder(IndexBuilder):
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = KnowledgeGraphIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index

class DocumentSummaryIndexBuilder(IndexBuilder):
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = DocumentSummaryIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = DocumentSummaryIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index

class MultiModelStoreIndexBuilder(IndexBuilder):
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = MultiModalVectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            **self.kwargs
        )
        return self._index



class IndexBuilderFactory:
    @staticmethod
    def create(builder_type: str,
               storage_context: Optional[StorageContext] = None,
               show_progress: bool = True,
               **kwargs) -> IndexBuilder:
        """インデックスビルダーを作成"""
        builder_type = builder_type.lower()
        builders = {
            "vector_store": VectorStoreIndexBuilder,
            "summary": SummaryIndexBuilder,
            "tree": TreeIndexBuilder,
            "keyword_table": KeywordTableIndexBuilder,
            "knowledge_graph": KnowledgeGraphIndexBuilder,
            "document_summary": DocumentSummaryIndexBuilder,
            "multi_model": MultiModelStoreIndexBuilder,
        }
        
        if builder_type not in builders:
            raise ValueError(f"Unknown evaluator type: {builder_type}")
        
        return builders[builder_type](
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs
        )
