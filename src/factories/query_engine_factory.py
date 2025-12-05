
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
from llama_index.core.tools import QueryEngineTool, RetrieverTool
from llama_index.core.indices.base import BaseIndex
from llama_index.core.response_synthesizers import TreeSummarize
from .template_prompts import TemplatePromptSettings
from .output_parser_factory import JapaneseSelectionOutputParser
from .response_synthesizer_factory import ResponseSynthesizerFactory, ResponseMode


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
    def create_retriever_query_engine(
        index: BaseIndex,
        retriever: BaseRetriever,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> RetrieverQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
        return RetrieverQueryEngine(
            index=index,
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
    
    @staticmethod
    def create_router_query_engine(
        selector_type: str,
        query_engine_tools: Sequence[QueryEngineTool],
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> RouterQueryEngine:
        selector = SelectorFactory.create(selector_type=selector_type)
        response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
        return RouterQueryEngine(
            selector=selector,
            query_engine_tools=query_engine_tools,
            summarizer=TreeSummarize(
                summary_template=TemplatePromptSettings.JP_TREE_SUMMARIZE_PROMPT_SEL,
                verbose=True
            )
        )
    
    # @staticmethod
    # def create_router_query_engine(
    #     selector_type: str,
    #     indices: List[Tuple[BaseIndex, str, str]],
    #     response_mode: ResponseMode = ResponseMode.COMPACT,
    # ) -> RouterQueryEngine:
    #     selector = SelectorFactory.create(selector_type=selector_type)
    #     response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
    #     return RouterQueryEngine(
    #         selector=selector,
    #         query_engine_tools=[ToolFactory.create_query_engine_tool(idx.as_query_engine(), name, desc) for idx, name, desc in indices],
    #         response_synthesizer=response_synthesizer
    #     )
        

    @staticmethod
    def create_retry_query_engine(
        query_engine: BaseQueryEngine,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> RetryQueryEngine:
        response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode)
        return RetryQueryEngine(
            query_engine=query_engine,
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
                prompt_template_str=TemplatePromptSettings.JP_SINGLE_SELECT_PROMPT_JSON_TMPL,
                output_parser=JapaneseSelectionOutputParser(),
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
                prompt_template_str=TemplatePromptSettings.JP_MULTI_SELECT_PROMPT_JSON_TMPL,
                output_parser=JapaneseSelectionOutputParser()
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