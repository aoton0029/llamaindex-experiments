
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
from llama_index.core.selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.indices.base import BaseIndex
from factories.response_synthesizer_service import ResponseSynthesizerFactory, ResponseMode


logger = logging.getLogger(__name__)


class QueryEngineFactory:
    @staticmethod
    def create(query_engine_type: str) -> BaseQueryEngine:
        if query_engine_type == "retriever":
            return RetrieverQueryEngine()
        elif query_engine_type == "router":
            return RouterQueryEngine()
        elif query_engine_type == "retry":
            return RetryQueryEngine()
        elif query_engine_type == "multi_step":
            return MultiStepQueryEngine()
        elif query_engine_type == "transform":
            return TransformQueryEngine()
        elif query_engine_type == "retry_source":
            return RetrySourceQueryEngine()
        else:
            raise ValueError(f"未知のクエリエンジンタイプ: {query_engine_type}")
    
    def create_retriever_query_engine(
        self,
        index: BaseIndex,
        retriever: BaseRetriever,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        **kwargs
    ) -> RetrieverQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.create(response_mode=response_mode)
        return RetrieverQueryEngine(
            index=index,
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
    
    def create_router_query_engine(
        self,
        selector_type: str,
        query_engine_tools: Sequence[QueryEngineTool],
        response_mode: ResponseMode = ResponseMode.DEFAULT,
    ) -> RouterQueryEngine:
        selector = SelectorFactory.create(selector_type=selector_type)
        response_synthesizer = ResponseSynthesizerFactory.create(response_mode=response_mode)
        return RouterQueryEngine(
            selector=selector,
            query_engine_tools=query_engine_tools,
            response_synthesizer=response_synthesizer
        )
    
    def create_router_query_engine(
        self,
        selector_type: str,
        indices: List[Tuple[BaseIndex, str, str]],
        response_mode: ResponseMode = ResponseMode.DEFAULT,
    ) -> RouterQueryEngine:
        selector = SelectorFactory.create(selector_type=selector_type)
        response_synthesizer = ResponseSynthesizerFactory.create(response_mode=response_mode)
        return RouterQueryEngine(
            selector=selector,
            query_engine_tools=[ToolFactory.create_query_engine_tool(idx.as_query_engine(), name, desc) for idx, name, desc in indices],
            response_synthesizer=response_synthesizer
        )

    def create_retry_query_engine(
        self,
        query_engine: BaseQueryEngine,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
    ) -> RetryQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.create(response_mode=response_mode)
        return RetryQueryEngine(
            query_engine=query_engine,
            response_synthesizer=response_synthesizer
        )
    
    def create_multi_step_query_engine(
        self,
        query_engines: List[BaseQueryEngine],
        response_mode: ResponseMode = ResponseMode.DEFAULT,
    ) -> MultiStepQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.create(response_mode=response_mode)
        return MultiStepQueryEngine(
            query_engines=query_engines,
            response_synthesizer=response_synthesizer
        )
    
    def create_transform_query_engine(
        self,
        query_engine: BaseQueryEngine,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
    ) -> TransformQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.create(response_mode=response_mode)
        return TransformQueryEngine(
            query_engine=query_engine,
            response_synthesizer=response_synthesizer
        )


class ToolFactory:
    @staticmethod
    def create_query_engine_tool(query_engine: BaseQueryEngine, name: str, description: str,) -> QueryEngineTool:
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
    def create_retriever_tool(retriever: BaseRetriever, name: str, description: str,) -> RetrieverTool:
        try:
            from llama_index.core.tools import RetrieverTool
            
            tool = RetrieverTool.from_defaults(
                retriever=retriever,
                name=name,
                description=description,
            )
            logger.info(f"RetrieverToolを作成: {name}")
            return tool
        except Exception as e:
            logger.error(f"RetrieverTool作成エラー: {e}")
            raise


class SelectorFactory:


    @staticmethod
    def create(selector_type:str):
        if selector_type == "llm_single":
            return SelectorFactory.create_llm_single_selector() 
        elif selector_type == "llm_multi":
            return SelectorFactory.create_llm_multi_selector()
        else:
            raise ValueError(f"未知のセレクタータイプ: {selector_type}")
    
    @staticmethod
    def create_llm_single_selector():
        try:
            selector = LLMSingleSelector.from_defaults(
                prompt_template_str=SelectorFactory.DEFAULT_SINGLE_SELECT_PROMPT_TMPL
            )
            logger.info("LLMSingleSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"LLMSingleSelector作成エラー: {e}")
            raise
    
    @staticmethod
    def create_llm_multi_selector():
        try:
            selector = LLMMultiSelector.from_defaults(
                prompt_template_str=SelectorFactory.DEFAULT_MULTI_SELECT_PROMPT_TMPL
            )
            logger.info("LLMMultiSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"LLMMultiSelector作成エラー: {e}")
            raise


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