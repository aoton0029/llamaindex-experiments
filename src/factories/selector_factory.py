import logging
from typing import Any, List, Dict, Optional
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.indices.base import BaseIndex
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.postprocessor.node import BaseNodePostprocessor
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.selectors.prompts import SingleSelectPrompt, MultiSelectPrompt

from .output_parser_factory import SelectionOutputParserJp
from .settings_template_prompts import TemplatePromptSettings
from .settings_llm import DomainLLMSettings

logger = logging.getLogger(__name__)


class LLMSingleSelectorJp(LLMSingleSelector):
    """日本語対応LLM Single Selector"""
    
    @classmethod
    def from_defaults(cls) -> "LLMSingleSelectorJp":
        """日本語デフォルト設定でインスタンスを作成"""
        llm = DomainLLMSettings.SELECTOR
        prompt_template_str = TemplatePromptSettings.SINGLE_SELECT_PROMPT_TMPL
        output_parser = SelectionOutputParserJp()
        
        prompt = SingleSelectPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SINGLE_SELECT,
        )
        
        logger.info("LLMSingleSelectorJp（日本語対応シングルセレクター）を作成")
        return cls(llm=llm, prompt=prompt)


class LLMMultiSelectorJp(LLMMultiSelector):
    """日本語対応LLM Multi Selector"""
    
    @classmethod
    def from_defaults(cls,max_outputs: Optional[int] = None) -> "LLMMultiSelectorJp":
        """日本語デフォルト設定でインスタンスを作成"""
        llm = DomainLLMSettings.SELECTOR
        prompt_template_str = TemplatePromptSettings.MULTI_SELECT_PROMPT_TMPL
        output_parser = SelectionOutputParserJp()
        prompt_template_str = output_parser.format(prompt_template_str)
        
        prompt = MultiSelectPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
           prompt_type=PromptType.MULTI_SELECT,
        )
        
        logger.info("LLMMultiSelectorJp（日本語対応マルチセレクター）を作成")
        return cls(llm=llm, prompt=prompt, max_outputs=max_outputs)