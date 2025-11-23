"""
Retriever Module
各種Retrieverの実装
"""

import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import (
    BaseRetriever,
    BaseImageRetriever,
    BasePGRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
    SummaryIndexRetriever,
    RouterRetriever,
    TreeRootRetriever,
    TransformRetriever,
    QueryFusionRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    TreeSelectLeafRetriever,
    SummaryIndexEmbeddingRetriever,
    VectorIndexAutoRetriever,
    KnowledgeGraphRAGRetriever
)
from llama_index.core.indices.base import BaseIndex

logger = logging.getLogger(__name__)

class GoldenRetriever(BaseRetriever):
    def __init__(self, index: BaseIndex, **kwargs: Any) -> None:
        super().__init__(index=index, **kwargs)


class RetrieverFactory:
    @staticmethod
    def create(retriever_type: str, index: BaseIndex, **kwargs) -> BaseRetriever:
        if retriever_type == "vector":
            return VectorIndexRetriever(index=index, **kwargs)
        elif retriever_type == "keyword_table":
            return KeywordTableSimpleRetriever(index=index, **kwargs)
        elif retriever_type == "summary":
            return SummaryIndexRetriever(index=index, **kwargs)
        elif retriever_type == "router":
            return RouterRetriever(index=index, **kwargs)
        elif retriever_type == "tree_root":
            return TreeRootRetriever(index=index, **kwargs)
        elif retriever_type == "transform":
            return TransformRetriever(index=index, **kwargs)
        elif retriever_type == "query_fusion":
            return QueryFusionRetriever(index=index, **kwargs)
        elif retriever_type == "auto_merging":
            return AutoMergingRetriever(index=index, **kwargs)
        elif retriever_type == "recursive":
            return RecursiveRetriever(index=index, **kwargs)
        elif retriever_type == "tree_select_leaf":
            return TreeSelectLeafRetriever(index=index, **kwargs)
        elif retriever_type == "summary_embedding":
            return SummaryIndexEmbeddingRetriever(index=index, **kwargs)
        elif retriever_type == "vector_auto":
            return VectorIndexAutoRetriever(index=index, **kwargs)
        elif retriever_type == "knowledge_graph_rag":
            return KnowledgeGraphRAGRetriever(index=index, **kwargs)
        else:
            raise ValueError(f"未知のリトリバータイプ: {retriever_type}")
    
