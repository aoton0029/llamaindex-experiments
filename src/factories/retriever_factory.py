import logging
from typing import List, Optional, Dict, Any, Sequence
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
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.indices.base import BaseIndex
from llama_index.core.postprocessor.node import BaseNodePostprocessor
from llama_index.core.selectors import BaseSelector
from llama_index.core.base.llms.base import BaseLLM

logger = logging.getLogger(__name__)

class GoldenRetriever(BaseRetriever):
    def __init__(self, index: BaseIndex, **kwargs: Any) -> None:
        super().__init__(index=index, **kwargs)


class RetrieverFactory:
    @staticmethod
    def create_retriever_tool(
        retriever: BaseRetriever, 
        name: str, 
        description: str,
        node_postprocessors: List[BaseNodePostprocessor] = None
    ) -> RetrieverTool:
        try:
            tool = RetrieverTool.from_defaults(
                retriever=retriever,
                name=name,
                description=description,
                node_postprocessors=node_postprocessors
            )
            logger.info(f"RetrieverToolを作成: {name}")
            return tool
        except Exception as e:
            logger.error(f"RetrieverTool作成エラー: {e}")
            raise
       
    @staticmethod
    def create_router_retriever(
        selector: BaseSelector,
        retriever_tools: Sequence[RetrieverTool],
        llm: BaseLLM,
    ) -> RouterRetriever:
        return RouterRetriever.from_defaults(
            selector=selector,
            retriever_tools=retriever_tools,
            llm=llm,
        )
    
