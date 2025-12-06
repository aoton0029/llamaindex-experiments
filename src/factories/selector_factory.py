import logging
from typing import Any, List, Dict, Optional
from llama_index.core.selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.core.indices.base import BaseIndex
from .output_parser_factory import JapaneseSelectionOutputParser
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.postprocessor.node import BaseNodePostprocessor
from .template_prompts import TemplatePromptSettings

logger = logging.getLogger(__name__)

class SelectorFactory:

    @staticmethod
    def create(selector_type:str, llm: BaseLLM, **kwargs):
        if selector_type == "llm_single":
            return SelectorFactory.create_llm_single_selector(llm=llm, **kwargs) 
        elif selector_type == "llm_multi":
            return SelectorFactory.create_llm_multi_selector(llm=llm, **kwargs)
        else:
            raise ValueError(f"未知のセレクタータイプ: {selector_type}")
    
    @staticmethod
    def create_llm_single_selector(llm: BaseLLM, **kwargs):
        try:
            selector = LLMSingleSelector.from_defaults(
                llm=llm,
                prompt_template_str=TemplatePromptSettings.JP_SINGLE_SELECT_PROMPT_JSON_TMPL,
                output_parser=JapaneseSelectionOutputParser(),
            )
            logger.info("LLMSingleSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"LLMSingleSelector作成エラー: {e}")
            raise
    
    @staticmethod
    def create_llm_multi_selector(llm: BaseLLM, max_outputs: int = 3):
        try:
            selector = LLMMultiSelector.from_defaults(
                llm=llm,
                prompt_template_str=TemplatePromptSettings.JP_MULTI_SELECT_PROMPT_JSON_TMPL,
                output_parser=JapaneseSelectionOutputParser(),
                max_outputs=max_outputs
            )
            logger.info("LLMMultiSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"LLMMultiSelector作成エラー: {e}")
            raise
