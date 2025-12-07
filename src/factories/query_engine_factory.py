import logging
from typing import List, Optional, Dict, Any, Sequence, Tuple
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
    RouterQueryEngine,
    RetryQueryEngine,
    MultiStepQueryEngine,
    TransformQueryEngine,
    RetrySourceQueryEngine
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.tools import QueryEngineTool
from .template_prompts import TemplatePromptSettings
from llama_index.core.response_synthesizers import TreeSummarize
from .response_synthesizer_factory import ResponseSynthesizerFactory, ResponseMode
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.postprocessor.node import BaseNodePostprocessor
from llama_index.core.selectors import BaseSelector


logger = logging.getLogger(__name__)


class QueryEngineFactory:
    @staticmethod
    def create(query_engine_type: str, **kwargs) -> BaseQueryEngine:
        if query_engine_type == "retriever":
            return QueryEngineFactory.create_retriever_query_engine(**kwargs)
        elif query_engine_type == "router":
            return QueryEngineFactory.create_router_query_engine(**kwargs)
        elif query_engine_type == "retry":
            return QueryEngineFactory.create_retry_query_engine(**kwargs)
        elif query_engine_type == "multi_step":
            return QueryEngineFactory.create_multi_step_query_engine(**kwargs)
        elif query_engine_type == "transform":
            return QueryEngineFactory.create_transform_query_engine(**kwargs)
        else:
            raise ValueError(f"未知のクエリエンジンタイプ: {query_engine_type}")
    
    @staticmethod
    def create_query_engine_tool(query_engine: BaseQueryEngine, name: str, description: str) -> QueryEngineTool:
        try:
            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=name,
                description=description,
            )
            logger.info(f"QueryEngineToolを作成: {name}")
            return tool
        except Exception as e:
            logger.error(f"QueryEngineTool作成エラー: {e}")
            raise
    
    @staticmethod
    def create_retriever_query_engine(
        retriever: BaseRetriever,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ) -> RetrieverQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors
        )
    
    @staticmethod
    def create_router_query_engine(
        selector: BaseSelector,
        query_engine_tools: Sequence[QueryEngineTool],
        query_engine_llm: BaseLLM,
        tree_summarize_llm: BaseLLM,
    ) -> RouterQueryEngine:
        tree_summarizer = TreeSummarize(
            llm=tree_summarize_llm,
            summary_template=TemplatePromptSettings.JP_TREE_SUMMARIZE_PROMPT_SEL,
            verbose=True
        )
        return RouterQueryEngine(
            selector=selector,
            query_engine_tools=query_engine_tools,
            llm=query_engine_llm,
            summarizer=tree_summarizer
        )

    @staticmethod
    def create_retry_query_engine(
        query_engine: BaseQueryEngine,
        evaluator: BaseEvaluator,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> RetryQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
        return RetryQueryEngine(
            query_engine=query_engine,
            evaluator=evaluator,
            response_synthesizer=response_synthesizer
        )
    
    
    @staticmethod
    def create_multi_step_query_engine(
        query_engines: List[BaseQueryEngine],
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> MultiStepQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
        return MultiStepQueryEngine(
            query_engines=query_engines,
            response_synthesizer=response_synthesizer
        )
    
    @staticmethod
    def create_transform_query_engine(
        query_engine: BaseQueryEngine,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> TransformQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
        return TransformQueryEngine(
            query_engine=query_engine,
            response_synthesizer=response_synthesizer
        )




class MetadataFilterFactory:
    @staticmethod
    def create_metadata_filter(
        key: str,
        value: Any,
        operator: str = "==",
        **kwargs
    ):
        try:
            from llama_index.core.vector_stores import MetadataFilter, FilterOperator
            
            # 演算子のマッピング
            operator_map = {
                "==": FilterOperator.EQ,
                "!=": FilterOperator.NE,
                ">": FilterOperator.GT,
                ">=": FilterOperator.GTE,
                "<": FilterOperator.LT,
                "<=": FilterOperator.LTE,
                "in": FilterOperator.IN,
                "nin": FilterOperator.NIN,
            }
            
            filter_operator = operator_map.get(operator, FilterOperator.EQ)
            
            metadata_filter = MetadataFilter(
                key=key,
                value=value,
                operator=filter_operator,
                **kwargs
            )
            logger.info(f"MetadataFilterを作成: {key} {operator} {value}")
            return metadata_filter
        except Exception as e:
            logger.error(f"MetadataFilter作成エラー: {e}")
            raise
    
    @staticmethod
    def create_metadata_filters(
        filters: List[Any],
        condition: str = "and",
        **kwargs
    ):
        try:
            from llama_index.core.vector_stores import MetadataFilters, FilterCondition
            
            filter_condition = FilterCondition.AND if condition == "and" else FilterCondition.OR
            
            metadata_filters = MetadataFilters(
                filters=filters,
                condition=filter_condition,
                **kwargs
            )
            logger.info(f"MetadataFiltersを作成: {len(filters)}個のフィルター")
            return metadata_filters
        except Exception as e:
            logger.error(f"MetadataFilters作成エラー: {e}")
            raise