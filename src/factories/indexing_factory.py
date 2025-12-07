import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from llama_index.core.schema import BaseNode, Document, TransformComponent
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
from llama_index.core.query_engine import BaseQueryEngine
from .template_prompts import TemplatePromptSettings
from .response_synthesizer_factory import ResponseSynthesizerFactory, ResponseMode
logger = logging.getLogger(__name__)


class IndexBuilder(ABC):
    def __init__(
        self,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = True,
    ):
        self.storage_context = storage_context
        self.show_progress = show_progress
    
    @abstractmethod
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        pass
    
    @abstractmethod
    def build_from_documents(self, documents: List[Document], transformations:List[TransformComponent] = None) -> BaseIndex:
        pass
    
    
class VectorStoreIndexBuilder(IndexBuilder):
    def __init__(self, 
                 storage_context = None, 
                 show_progress = True):
        super().__init__(storage_context, show_progress)
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            store_nodes_override=True,
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            store_nodes_override=True,
        )
        return self._index


class SummaryIndexBuilder(IndexBuilder):
    """

    1. SummaryIndexの特徴
        - クエリ時に動的に要約を生成する
        - インデックス作成時には要約を事前計算せず、ドキュメントのメタデータのみを保存する
    2. 保存される内容
        - docstore: ドキュメントの本文とメタデータ
        - index_store: インデックス構造（doc_idのリストなど）
    """
    def __init__(self, 
                 storage_context = None, 
                 show_progress = True):
        super().__init__(storage_context, show_progress)
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = SummaryIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            text_qa_template=TemplatePromptSettings.JP_TEXT_QA_PROMPT
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = SummaryIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            text_qa_template=TemplatePromptSettings.JP_TEXT_QA_PROMPT
        )
        return self._index

class TreeIndexBuilder(IndexBuilder):
    """

    1. TreeIndexの特徴
        - インデックス作成時に生成される
        - 階層的なツリー構造と各ノードの要約
    2. 保存される内容
        - index_store: ツリー構造
        - doc_store: 要約
    """
    def __init__(self, 
                 llm = None,
                 storage_context = None, 
                 show_progress = True):
        super().__init__(storage_context, show_progress)
        self.llm = llm

    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = TreeIndex(
            nodes,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            summary_template=TemplatePromptSettings.JP_SUMMARY_PROMPT,
            insert_prompt=TemplatePromptSettings.JP_INSERT_PROMPT,
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = TreeIndex.from_documents(
            documents,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            summary_template=TemplatePromptSettings.JP_SUMMARY_PROMPT,
            insert_prompt=TemplatePromptSettings.JP_INSERT_PROMPT,
        )
        return self._index

class KeywordTableIndexBuilder(IndexBuilder):
    """

    1. KeywordTableIndexの特徴
        - インデックス生成時にキーワードを生成
    2. 保存される内容
        - index_store: キーワード → doc_idのマッピング
    """
    def __init__(self, 
                 llm = None,
                 storage_context = None, 
                 show_progress = True, 
                 max_keywords_per_chunk=10):
        super().__init__(storage_context, show_progress)
        self.llm = llm
        self.max_keywords_per_chunk = max_keywords_per_chunk


    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = KeywordTableIndex(
            nodes,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            keyword_extract_template=TemplatePromptSettings.JP_KEYWORD_EXTRACT_TEMPLATE,
            max_keywords_per_chunk=self.max_keywords_per_chunk
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = KeywordTableIndex.from_documents(
            documents,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            keyword_extract_template=TemplatePromptSettings.JP_KEYWORD_EXTRACT_TEMPLATE,
            max_keywords_per_chunk=self.max_keywords_per_chunk
        )
        return self._index

class KnowledgeGraphIndexBuilder(IndexBuilder):
    def __init__(self, 
                 llm = None,
                 storage_context = None, 
                 show_progress = True):
        super().__init__(storage_context, show_progress)
        self.llm = llm
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = KnowledgeGraphIndex(
            nodes,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = KnowledgeGraphIndex.from_documents(
            documents,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
        )
        return self._index

class DocumentSummaryIndexBuilder(IndexBuilder):
    """

    1. DocumentSummaryIndexの特徴
        - インデックス生成時のみ要約を生成する
    2. 保存される内容
        - docstore: 元のドキュメント、各ドキュメントの要約
        - index_store: インデックス構造
    """
    def __init__(self, 
                 llm = None,
                 storage_context = None, 
                 show_progress = True):
        super().__init__(storage_context, show_progress)
        self.llm = llm

    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = DocumentSummaryIndex(
            nodes,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            response_synthesizer=ResponseSynthesizerFactory.get(ResponseMode.TREE_SUMMARIZE),
            summary_query=TemplatePromptSettings.JP_SUMMARY_QUERY,
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = DocumentSummaryIndex.from_documents(
            documents,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
            response_synthesizer=ResponseSynthesizerFactory.get(ResponseMode.TREE_SUMMARIZE),
            summary_query=TemplatePromptSettings.JP_SUMMARY_QUERY,
        )
        return self._index

class MultiModelStoreIndexBuilder(IndexBuilder):
    def __init__(self, 
                 storage_context = None, 
                 show_progress = True):
        super().__init__(storage_context, show_progress)
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        self._index = MultiModalVectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
        )
        return self._index
    
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        self._index = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=self.show_progress,
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
            "vector": VectorStoreIndexBuilder,
            "vector_store": VectorStoreIndexBuilder,
            "summary": SummaryIndexBuilder,
            "tree": TreeIndexBuilder,
            "keyword_table": KeywordTableIndexBuilder,
            "knowledge_graph": KnowledgeGraphIndexBuilder,
            "document_summary": DocumentSummaryIndexBuilder,
            "multi_model": MultiModelStoreIndexBuilder,
        }
        
        if builder_type not in builders:
            raise ValueError(f"Unknown builder type: {builder_type}")
        
        return builders[builder_type](
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs
        )

