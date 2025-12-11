from typing import Any, Optional, Dict
from llama_index.core import Settings
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class _TemplatePromptSettings:
    """テンプレートプロンプト設定クラス"""
    
    _prompt_tmpls: Optional[Dict[str, Any]] = None
    _templates: Dict[str, Any] = {}
    _templates_loaded: bool = False

    def initialize(self, prompt_tmpls: Optional[Dict[str, Any]] = None):
        """テンプレートを初期化
        
        Args:
            prompt_tmpls: YAMLから読み込まれたtemplate_promptsの辞書
        """
        self._templates_loaded = False
        self._prompt_tmpls = prompt_tmpls
        self._templates = {}  # 辞書をリセット
        self._load_templates()

    def _load_templates(self):
        """YAMLからテンプレート文字列のみを読み込み、辞書に格納"""
        if self._templates_loaded or self._prompt_tmpls is None:
            return
        
        try:
            config = self._prompt_tmpls

            # Simple Input
            if "simple_input" in config:
                simple_input = config["simple_input"]
                if "JP_SIMPLE_INPUT_TMPL" in simple_input:
                    self._templates["JP_SIMPLE_INPUT_TMPL"] = simple_input["JP_SIMPLE_INPUT_TMPL"]
            
            # Text QA
            if "text_qa" in config:
                text_qa = config["text_qa"]
                if "CHAT_SYSTEM_PROMPT" in text_qa:
                    self._templates["TEXT_QA_SYSTEM_PROMPT_TMPL"] = text_qa["CHAT_SYSTEM_PROMPT"]
                if "CHAT_USER_PROMPT" in text_qa:
                    self._templates["TEXT_QA_USER_PROMPT_TMPL"] = text_qa["CHAT_USER_PROMPT"]
                if "JP_TEXT_QA_TMPL" in text_qa:
                    self._templates["JP_TEXT_QA_TMPL"] = text_qa["JP_TEXT_QA_TMPL"]
            
            # Tree Summarize
            if "tree_summarize" in config:
                tree_sum = config["tree_summarize"]
                if "CHAT_SYSTEM_PROMPT" in tree_sum:
                    self._templates["TREE_SUMMARIZE_SYSTEM_PROMPT_TMPL"] = tree_sum["CHAT_SYSTEM_PROMPT"]
                if "CHAT_USER_PROMPT" in tree_sum:
                    self._templates["TREE_SUMMARIZE_USER_PROMPT_TMPL"] = tree_sum["CHAT_USER_PROMPT"]
                if "JP_TREE_SUMMARIZE_TMPL" in tree_sum:
                    self._templates["JP_TREE_SUMMARIZE_TMPL"] = tree_sum["JP_TREE_SUMMARIZE_TMPL"]
            
            # Refine
            if "refine" in config:
                refine = config["refine"]
                if "CHAT_USER_PROMPT" in refine:
                    self._templates["REFINE_USER_PROMPT_TMPL"] = refine["CHAT_USER_PROMPT"]
                if "JP_REFINE_TMPL" in refine:
                    self._templates["JP_REFINE_TMPL"] = refine["JP_REFINE_TMPL"]
            
            # Refine Table Context
            if "refine_table_context" in config:
                refine_table = config["refine_table_context"]
                if "CHAT_QUERY_PROMPT" in refine_table:
                    self._templates["REFINE_TABLE_QUERY_PROMPT_TMPL"] = refine_table["CHAT_QUERY_PROMPT"]
                if "CHAT_EXISTING_ANSWER_PROMPT" in refine_table:
                    self._templates["REFINE_TABLE_EXISTING_ANSWER_TMPL"] = refine_table["CHAT_EXISTING_ANSWER_PROMPT"]
                if "CHAT_CONTEXT_PROMPT" in refine_table:
                    self._templates["REFINE_TABLE_CONTEXT_PROMPT_TMPL"] = refine_table["CHAT_CONTEXT_PROMPT"]
                if "JP_REFINE_TABLE_CONTEXT_TMPL" in refine_table:
                    self._templates["JP_REFINE_TABLE_CONTEXT_TMPL"] = refine_table["JP_REFINE_TABLE_CONTEXT_TMPL"]
            
            # Selection
            if "selection" in config:
                selection = config["selection"]
                for key in ["JP_SINGLE_SELECT_TMPL", 
                            "JP_SINGLE_SELECT_JSON_TMPL", 
                            "JP_MULTI_SELECT_TMPL", 
                            "JP_MULTI_SELECT_JSON_TMPL", 
                            "JP_SELECTION_FORMAT_JSON_TMPL"]:
                    if key in selection:
                        self._templates[key] = selection[key]
            
            # Extractor
            if "extractor" in config:
                extractor = config["extractor"]
                mapping = {
                    "JP_TITLE_NODE_TMPL": "JP_TITLE_NODE_TMPL",
                    "JP_TITLE_COMBINE_TMPL": "JP_TITLE_COMBINE_TMPL",
                    "JP_SUMMARY_EXTRACT_TMPL": "JP_SUMMARY_EXTRACT_TMPL",
                    "JP_KEYWORD_EXTRACT_TMPL": "JP_KEYWORD_EXTRACT_TMPL",
                    "JP_QUESTION_GEN_TMPL": "JP_QUESTION_GEN_TMPL"
                }
                for yaml_key, template_key in mapping.items():
                    if yaml_key in extractor:
                        self._templates[template_key] = extractor[yaml_key]
            
            # Evaluation
            if "evaluation" in config:
                evaluation = config["evaluation"]
                for key in ["JP_QUESTION_GENERATION_PROMPT", 
                           "JP_QUESTION_GEN_QUERY", 
                           "JP_SUMMARY_QUERY",
                           "JP_EVAL_TMPL",
                           "JP_REFINE_EVAL_TMPL",
                           "JP_SUMMARY_PROMPT_TMPL",
                           "JP_INSERT_PROMPT_TMPL"]:
                    if key in evaluation:
                        self._templates[key] = evaluation[key]
            
            self._templates_loaded = True
            logger.info(f"Loaded {len(self._templates)} template strings from YAML")
        except Exception as e:
            logger.warning(f"Failed to load templates from YAML: {e}")
            self._templates = {}
    
    
    def get(self, key: str, default: Any = None) -> Any:
        """テンプレートを取得（辞書アクセス）"""
        if not self._templates_loaded and self._prompt_tmpls:
            self._load_templates()
        return self._templates.get(key, default)
    
    # Simple Input
    @property
    def SIMPLE_INPUT_PROMPT(self) -> Optional[PromptTemplate]:
        """Simple Input用プロンプト"""
        tmpl = self.get("JP_SIMPLE_INPUT_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.SIMPLE_INPUT) if tmpl else None
    
    # Text QA
    @property
    def TEXT_QA_PROMPT(self) -> Optional[PromptTemplate]:
        """Text QA用プロンプト（日本語）"""
        tmpl = self.get("JP_TEXT_QA_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.QUESTION_ANSWER) if tmpl else None
    
    @property
    def TEXT_QA_PROMPT_SEL(self) -> Optional[SelectorPromptTemplate]:
        """Text QA用Selectorプロンプト"""
        jp_prompt = self.TEXT_QA_PROMPT
        chat_prompt = self.CHAT_TEXT_QA_PROMPT
        if jp_prompt and chat_prompt:
            conditionals = [(self.is_chat_model, chat_prompt)]
            return SelectorPromptTemplate(default_template=jp_prompt, conditionals=conditionals)
        return None
    
    @property
    def CHAT_TEXT_QA_PROMPT(self) -> Optional[ChatPromptTemplate]:
        """Text QA用Chatプロンプト"""
        sys_tmpl = self.get("TEXT_QA_SYSTEM_PROMPT_TMPL")
        user_tmpl = self.get("TEXT_QA_USER_PROMPT_TMPL")
        if sys_tmpl and user_tmpl:
            msgs = [
                ChatMessage(content=sys_tmpl, role=MessageRole.SYSTEM),
                ChatMessage(content=user_tmpl, role=MessageRole.USER)
            ]
            return ChatPromptTemplate(message_templates=msgs)
        return None
    
    # Tree Summarize
    @property
    def TREE_SUMMARIZE_PROMPT_TMPL(self) -> Optional[str]:
        """Tree Summarize用テンプレート文字列"""
        return self.get("JP_TREE_SUMMARIZE_TMPL")

    @property
    def TREE_SUMMARIZE_PROMPT(self) -> Optional[PromptTemplate]:
        """Tree Summarize用プロンプト（日本語）"""
        tmpl = self.get("JP_TREE_SUMMARIZE_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.SUMMARY) if tmpl else None
    
    @property
    def TREE_SUMMARIZE_PROMPT_SEL(self) -> Optional[SelectorPromptTemplate]:
        """Tree Summarize用Selectorプロンプト"""
        jp_prompt = self.TREE_SUMMARIZE_PROMPT
        chat_prompt = self.CHAT_TREE_SUMMARIZE_PROMPT
        if jp_prompt and chat_prompt:
            conditionals = [(self.is_chat_model, chat_prompt)]
            return SelectorPromptTemplate(default_template=jp_prompt, conditionals=conditionals)
        return None
    
    @property
    def CHAT_TREE_SUMMARIZE_PROMPT(self) -> Optional[ChatPromptTemplate]:
        """Tree Summarize用Chatプロンプト"""
        sys_tmpl = self.get("TREE_SUMMARIZE_SYSTEM_PROMPT_TMPL")
        user_tmpl = self.get("TREE_SUMMARIZE_USER_PROMPT_TMPL")
        if sys_tmpl and user_tmpl:
            msgs = [
                ChatMessage(content=sys_tmpl, role=MessageRole.SYSTEM),
                ChatMessage(content=user_tmpl, role=MessageRole.USER)
            ]
            return ChatPromptTemplate(message_templates=msgs)
        return None
    
    # Refine
    @property
    def REFINE_PROMPT(self) -> Optional[PromptTemplate]:
        """Refine用プロンプト（日本語）"""
        tmpl = self.get("JP_REFINE_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.REFINE) if tmpl else None
    
    @property
    def REFINE_PROMPT_SEL(self) -> Optional[SelectorPromptTemplate]:
        """Refine用Selectorプロンプト"""
        jp_prompt = self.REFINE_PROMPT
        chat_prompt = self.CHAT_REFINE_PROMPT
        if jp_prompt and chat_prompt:
            conditionals = [(self.is_chat_model, chat_prompt)]
            return SelectorPromptTemplate(default_template=jp_prompt, conditionals=conditionals)
        return None
    
    @property
    def CHAT_REFINE_PROMPT(self) -> Optional[ChatPromptTemplate]:
        """Refine用Chatプロンプト"""
        user_tmpl = self.get("REFINE_USER_PROMPT_TMPL")
        if user_tmpl:
            msgs = [ChatMessage(content=user_tmpl, role=MessageRole.USER)]
            return ChatPromptTemplate(message_templates=msgs)
        return None
    
    # Refine Table Context
    @property
    def REFINE_TABLE_CONTEXT_PROMPT(self) -> Optional[PromptTemplate]:
        """Refine Table Context用プロンプト（日本語）"""
        tmpl = self.get("JP_REFINE_TABLE_CONTEXT_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.TABLE_CONTEXT) if tmpl else None
    
    @property
    def REFINE_TABLE_CONTEXT_PROMPT_SEL(self) -> Optional[SelectorPromptTemplate]:
        """Refine Table Context用Selectorプロンプト"""
        jp_prompt = self.REFINE_TABLE_CONTEXT_PROMPT
        chat_prompt = self.CHAT_REFINE_TABLE_CONTEXT_PROMPT
        if jp_prompt and chat_prompt:
            conditionals = [(self.is_chat_model, chat_prompt)]
            return SelectorPromptTemplate(default_template=jp_prompt, conditionals=conditionals)
        return None
    
    @property
    def CHAT_REFINE_TABLE_CONTEXT_PROMPT(self) -> Optional[ChatPromptTemplate]:
        """Refine Table Context用Chatプロンプト"""
        query_tmpl = self.get("REFINE_TABLE_QUERY_PROMPT_TMPL")
        answer_tmpl = self.get("REFINE_TABLE_EXISTING_ANSWER_TMPL")
        context_tmpl = self.get("REFINE_TABLE_CONTEXT_PROMPT_TMPL")
        if query_tmpl and answer_tmpl and context_tmpl:
            msgs = [
                ChatMessage(content=query_tmpl, role=MessageRole.USER),
                ChatMessage(content=answer_tmpl, role=MessageRole.ASSISTANT),
                ChatMessage(content=context_tmpl, role=MessageRole.USER)
            ]
            return ChatPromptTemplate(message_templates=msgs)
        return None
    
    # Selection
    @property
    def SINGLE_SELECT_PROMPT_TMPL(self) -> Optional[str]:
        """Single Select用テンプレート文字列"""
        return self.get("JP_SINGLE_SELECT_TMPL")
    
    @property
    def MULTI_SELECT_PROMPT_TMPL(self) -> Optional[str]:
        """Multi Select用テンプレート文字列"""
        return self.get("JP_MULTI_SELECT_TMPL")
    
    # Extractor
    @property
    def TITLE_NODE_TMPL(self) -> Optional[str]:
        """Title Node用テンプレート文字列"""
        return self.get("JP_TITLE_NODE_TMPL")
    
    @property
    def TITLE_COMBINE_TMPL(self) -> Optional[str]:
        """Title Combine用テンプレート文字列"""
        return self.get("JP_TITLE_COMBINE_TMPL")
    
    @property
    def SUMMARY_EXTRACT_TMPL(self) -> Optional[str]:
        """Summary Extract用テンプレート文字列"""
        return self.get("JP_SUMMARY_EXTRACT_TMPL")
    
    @property
    def KEYWORD_EXTRACT_TEMPLATE(self) -> Optional[PromptTemplate]:
        """Keyword Extract用プロンプト"""
        tmpl = self.get("JP_KEYWORD_EXTRACT_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.KEYWORD_EXTRACT) if tmpl else None

    @property    
    def KEYWORD_EXTRACT_TEMPLATE_TMPL(self) -> str:
        """Keyword Extract用テンプレート文字列"""
        return self.get("JP_KEYWORD_EXTRACT_TEMPLATE_TMPL")

    @property
    def QUESTION_GEN_TMPL(self) -> Optional[str]:
        """Question Generation用テンプレート文字列"""
        return self.get("JP_QUESTION_GEN_TMPL")
    
    # Evaluation
    @property
    def EVAL_TEMPLATE(self) -> Optional[PromptTemplate]:
        """Evaluation用プロンプト"""
        tmpl = self.get("JP_EVAL_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.SIMPLE_INPUT) if tmpl else None
    
    @property
    def SUMMARY_PROMPT(self) -> Optional[PromptTemplate]:
        """Summary用プロンプト"""
        tmpl = self.get("JP_SUMMARY_PROMPT_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.SUMMARY) if tmpl else None
    
    @property
    def SUMMARY_QUERY_TMPL(self) -> str:
        """Summary Query用プロンプト"""
        return self.get("JP_SUMMARY_QUERY_TMPL")

    @property
    def INSERT_PROMPT(self) -> Optional[PromptTemplate]:
        """Tree Insert用プロンプト"""
        tmpl = self.get("JP_INSERT_PROMPT_TMPL")
        return PromptTemplate(tmpl, prompt_type=PromptType.TREE_INSERT) if tmpl else None
    
    def get_templates_info(self) -> dict:
        """現在のテンプレート設定情報を取得（テスト記録用）"""
        if not self._templates_loaded and self._prompt_tmpls:
            self._load_templates()
        
        return {
            "templates_loaded_from_yaml": self._templates_loaded,
            "loaded_template_count": len(self._templates),
            "available_templates": list(self._templates.keys()),
        }
    
    @staticmethod
    def is_chat_model() -> bool:
        """現在のLLMがチャットモデルかどうかを判定"""
        return Settings.llm.metadata.is_chat_model


TemplatePromptSettings = _TemplatePromptSettings()