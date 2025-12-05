from dataclasses import dataclass
from typing import Any, Optional, Dict
from llama_index.core import Settings
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate


class _TemplatePromptSettings:
    """テンプレートプロンプト設定クラス
    
    デフォルトはNone（LlamaIndexのデフォルトを使用）。
    YAMLファイルから設定を読み込んでオーバーライドできる。
    """
        
    # Simple Input
    JP_SIMPLE_INPUT_TMPL: str = None
    JP_SIMPLE_INPUT_PROMPT: PromptTemplate = None

    # Text QA
    TEXT_QA_SYSTEM_PROMPT_TMPL: str = None
    TEXT_QA_USER_PROMPT_TMPL: str = None
    TEXT_QA_SYSTEM_PROMPT: PromptTemplate = None
    TEXT_QA_PROMPT_TMPL_MSGS = None
    CHAT_TEXT_QA_PROMPT: PromptTemplate = None
    JP_TEXT_QA_PROMPT_TMPL: str = None
    JP_TEXT_QA_PROMPT: PromptTemplate = None
    default_text_qa_conditionals = None
    JP_TEXT_QA_PROMPT_SEL: SelectorPromptTemplate = None
    # Tree Summarize
    TREE_SUMMARIZE_SYSTEM_PROMPT_TMPL: str = None
    TREE_SUMMARIZE_USER_PROMPT_TMPL: str = None
    TREE_SUMMARIZE_SYSTEM_PROMPT: PromptTemplate = None
    TREE_SUMMARIZE_PROMPT_TMPL_MSGS = None
    CHAT_TREE_SUMMARIZE_PROMPT: PromptTemplate = None
    JP_TREE_SUMMARIZE_TMPL: str = None
    JP_TREE_SUMMARIZE_PROMPT: PromptTemplate = None
    default_tree_summarize_conditionals = None
    JP_TREE_SUMMARIZE_PROMPT_SEL: SelectorPromptTemplate = None
    # Refine Prompt
    CHAT_REFINE_USER_PROMPT_TMPL: str = None
    CHAT_REFINE_PROMPT_TMPL_MSGS = None
    CHAT_REFINE_PROMPT: PromptTemplate = None
    JP_REFINE_PROMPT_TMPL: str = None
    JP_REFINE_PROMPT: PromptTemplate = None
    default_refine_conditionals = None
    JP_REFINE_PROMPT_SEL: SelectorPromptTemplate = None
    # Refine Table Context
    CHAT_REFINE_TABLE_QUERY_PROMPT_TMPL: str = None
    CHAT_REFINE_TABLE_EXISTING_ANSWER_TMPL: str = None
    CHAT_REFINE_TABLE_CONTEXT_PROMPT_TMPL: str = None
    CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = None
    CHAT_REFINE_TABLE_CONTEXT_PROMPT: PromptTemplate = None
    JP_REFINE_TABLE_CONTEXT_TMPL: str = None
    JP_REFINE_TABLE_CONTEXT_PROMPT: PromptTemplate = None
    default_refine_table_conditionals = None
    JP_REFINE_TABLE_CONTEXT_PROMPT_SEL: SelectorPromptTemplate = None
    # Selection
    JP_SINGLE_SELECT_PROMPT_TMPL: str = None
    JP_MULTI_SELECT_PROMPT_TMPL: str = None
    # Extractor
    JP_TITLE_NODE_TMPL: str = None
    JP_TITLE_COMBINE_TMPL: str = None
    JP_SUMMARY_EXTRACT_TMPL: str = None
    JP_QUESTION_GEN_TMPL: str = None
    # Evaluation
    JP_QUESTION_GENERATION_PROMPT: PromptTemplate = None
    JP_QUESTION_GEN_QUERY: str = None
    JP_EVAL_TEMPLATE_TMPL: str = None
    JP_EVAL_TEMPLATE: PromptTemplate = None
    JP_REFINE_TEMPLATE_TMPL: str = None
    JP_REFINE_TEMPLATE: PromptTemplate = None
    JP_SUMMARY_QUERY = None
    JP_SUMMARY_PROMPT_TMPL: str = None
    JP_SUMMARY_PROMPT: PromptTemplate = None
    JP_INSERT_PROMPT_TMPL: str = None
    JP_INSERT_PROMPT: PromptTemplate = None
    JP_KEYWORD_EXTRACT_TEMPLATE_TMPL: str = None
    JP_KEYWORD_EXTRACT_TEMPLATE: PromptTemplate = None
    
    _configs: Optional[Dict[str, Any]] = None
    _templates_loaded: bool = False
    
    @classmethod
    def initialize(cls, configs: Optional[Dict[str, Any]] = None):
        """ConfigManagerを設定してテンプレートを初期化"""
        cls._templates_loaded = False
        cls._configs = configs
        cls._load_templates()

    @classmethod
    def _load_templates(cls):
        """YAMLからテンプレートを読み込み"""
        if cls._templates_loaded or cls._configs is None:
            return
        
        try:
            templates = cls._configs#.get("template_prompts", {})
            # 各カテゴリのテンプレートをオーバーライド
            if "simple_input" in templates:
                simple_input = templates["simple_input"]
                if "jp_simple_input_tmpl" in simple_input:
                    cls.JP_SIMPLE_INPUT_TMPL = simple_input["jp_simple_input_tmpl"]
                
            if "text_qa" in templates:
                text_qa = templates["text_qa"]
                if "chat_system_prompt" in text_qa:
                    cls.TEXT_QA_SYSTEM_PROMPT_TMPL = text_qa["chat_system_prompt"]
                if "chat_user_prompt" in text_qa:
                    cls.TEXT_QA_USER_PROMPT_TMPL = text_qa["chat_user_prompt"]
                if "jp_text_qa_tmpl" in text_qa:
                    cls.JP_TEXT_QA_PROMPT_TMPL = text_qa["jp_text_qa_tmpl"]
            
            if "tree_summarize" in templates:
                tree_summarize = templates["tree_summarize"]
                if "chat_system_prompt" in tree_summarize:
                    cls.TREE_SUMMARIZE_SYSTEM_PROMPT_TMPL = tree_summarize["chat_system_prompt"]
                if "chat_user_prompt" in tree_summarize:
                    cls.TREE_SUMMARIZE_USER_PROMPT_TMPL = tree_summarize["chat_user_prompt"]
                if "jp_tree_summarize_tmpl" in tree_summarize:
                    cls.JP_TREE_SUMMARIZE_TMPL = tree_summarize["jp_tree_summarize_tmpl"]
            
            if "refine" in templates:
                refine = templates["refine"]
                if "chat_user_prompt" in refine:
                    cls.CHAT_REFINE_USER_PROMPT_TMPL = refine["chat_user_prompt"]
                if "jp_refine_tmpl" in refine:
                    cls.JP_REFINE_PROMPT_TMPL = refine["jp_refine_tmpl"]
            
            if "refine_table_context" in templates:
                refine_table = templates["refine_table_context"]
                if "chat_query_prompt" in refine_table:
                    cls.CHAT_REFINE_TABLE_QUERY_PROMPT_TMPL = refine_table["chat_query_prompt"]
                if "chat_existing_answer_prompt" in refine_table:
                    cls.CHAT_REFINE_TABLE_EXISTING_ANSWER_TMPL = refine_table["chat_existing_answer_prompt"]
                if "chat_context_prompt" in refine_table:
                    cls.CHAT_REFINE_TABLE_CONTEXT_PROMPT_TMPL = refine_table["chat_context_prompt"]
                if "jp_refine_table_context_tmpl" in refine_table:
                    cls.JP_REFINE_TABLE_CONTEXT_TMPL = refine_table["jp_refine_table_context_tmpl"]
            
            if "selection" in templates:
                selection = templates["selection"]
                if "jp_single_select_tmpl" in selection:
                    cls.JP_SINGLE_SELECT_PROMPT_TMPL = selection["jp_single_select_tmpl"]
                if "jp_multi_select_tmpl" in selection:
                    cls.JP_MULTI_SELECT_PROMPT_TMPL = selection["jp_multi_select_tmpl"]
            
            if "extractor" in templates:
                extractor = templates["extractor"]
                if "jp_title_node_template_tmpl" in extractor:
                    cls.JP_TITLE_NODE_TMPL = extractor["jp_title_node_template_tmpl"]
                if "jp_title_combine_template_tmpl" in extractor:
                    cls.JP_TITLE_COMBINE_TMPL = extractor["jp_title_combine_template_tmpl"]
                if "jp_summary_extract_template_tmpl" in extractor:
                    cls.JP_SUMMARY_EXTRACT_TMPL = extractor["jp_summary_extract_template_tmpl"]
                if "jp_keyword_extract_template_tmpl" in extractor:
                    cls.JP_KEYWORD_EXTRACT_TEMPLATE_TMPL = extractor["jp_keyword_extract_template_tmpl"]
                if "jp_question_gen_tmpl" in extractor:
                    cls.JP_QUESTION_GEN_TMPL = extractor["jp_question_gen_tmpl"]
            
            if "evaluation" in templates:
                evaluation = templates["evaluation"]
                if "jp_question_generation_prompt" in evaluation:
                    cls.JP_QUESTION_GENERATION_PROMPT = evaluation["jp_question_generation_prompt"]
                if "jp_question_gen_query" in evaluation:
                    cls.JP_QUESTION_GEN_QUERY = evaluation["jp_question_gen_query"]
                if "jp_eval_template_tmpl" in evaluation:
                    cls.JP_EVAL_TEMPLATE_TMPL = evaluation["jp_eval_template_tmpl"]
                if "jp_refine_template_tmpl" in evaluation:
                    cls.JP_REFINE_TEMPLATE_TMPL = evaluation["jp_refine_template_tmpl"]
                if "jp_summary_query" in evaluation:
                    cls.JP_SUMMARY_QUERY = evaluation["jp_summary_query"]
                if "jp_summary_prompt_tmpl" in evaluation:
                    cls.JP_SUMMARY_PROMPT_TMPL = evaluation["jp_summary_prompt_tmpl"]
                if "jp_insert_prompt_tmpl" in evaluation:
                    cls.JP_INSERT_PROMPT_TMPL = evaluation["jp_insert_prompt_tmpl"]
            
            # プロンプトオブジェクトを再生成
            cls._regenerate_prompts()
            
            cls._templates_loaded = True
        except Exception as e:
            # エラーが発生してもデフォルト値（None）を使用
            print(f"Warning: Failed to load templates from YAML: {e}")
    
    @classmethod
    def _regenerate_prompts(cls):
        """テンプレート文字列からプロンプトオブジェクトを再生成（Noneでない場合のみ）"""
        if cls.TEXT_QA_SYSTEM_PROMPT_TMPL is not None and cls.TEXT_QA_USER_PROMPT_TMPL:
            cls.TEXT_QA_SYSTEM_PROMPT = ChatMessage(
                content=cls.TEXT_QA_SYSTEM_PROMPT_TMPL,
                role=MessageRole.SYSTEM
            )
            cls.TEXT_QA_PROMPT_TMPL_MSGS = [
                cls.TEXT_QA_SYSTEM_PROMPT,
                ChatMessage(
                    content=cls.TEXT_QA_USER_PROMPT_TMPL,
                    role=MessageRole.USER
                )
            ]
            cls.CHAT_TEXT_QA_PROMPT =ChatPromptTemplate(message_templates=cls.TEXT_QA_PROMPT_TMPL_MSGS)
            cls.default_text_qa_conditionals = [(cls.is_chat_model, cls.CHAT_TEXT_QA_PROMPT)]
        
        if cls.TREE_SUMMARIZE_SYSTEM_PROMPT_TMPL is not None and cls.TREE_SUMMARIZE_USER_PROMPT_TMPL is not None:
            cls.TREE_SUMMARIZE_SYSTEM_PROMPT = ChatMessage(
                content=cls.TREE_SUMMARIZE_SYSTEM_PROMPT_TMPL,
                role=MessageRole.SYSTEM,
            )
            cls.TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
                cls.TREE_SUMMARIZE_SYSTEM_PROMPT,
                ChatMessage(
                    content=cls.TREE_SUMMARIZE_USER_PROMPT_TMPL,
                    role=MessageRole.USER,
                ),
            ]
            cls.CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
                message_templates=cls.TREE_SUMMARIZE_PROMPT_TMPL_MSGS
            )
            cls.default_tree_summarize_conditionals = [(cls.is_chat_model, cls.CHAT_TREE_SUMMARIZE_PROMPT)]
        
        if cls.CHAT_REFINE_USER_PROMPT_TMPL:
            cls.CHAT_REFINE_PROMPT_TMPL_MSGS = [
                ChatMessage(
                    content=cls.CHAT_REFINE_USER_PROMPT_TMPL,
                    role=MessageRole.USER,
                )
            ]
            cls.CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=cls.CHAT_REFINE_PROMPT_TMPL_MSGS)
            cls.default_refine_conditionals = [(cls.is_chat_model, cls.CHAT_REFINE_PROMPT)]
        
        if cls.CHAT_REFINE_TABLE_QUERY_PROMPT_TMPL is not None and cls.CHAT_REFINE_TABLE_EXISTING_ANSWER_TMPL is not None and cls.CHAT_REFINE_TABLE_CONTEXT_PROMPT_TMPL is not None:
            cls.CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
                ChatMessage(content=cls.CHAT_REFINE_TABLE_QUERY_PROMPT_TMPL, role=MessageRole.USER),
                ChatMessage(content=cls.CHAT_REFINE_TABLE_EXISTING_ANSWER_TMPL, role=MessageRole.ASSISTANT),
                ChatMessage(
                    content=cls.CHAT_REFINE_TABLE_CONTEXT_PROMPT_TMPL,
                    role=MessageRole.USER,
                ),
            ]
            cls.CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
                message_templates=cls.CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
            )
            cls.default_refine_table_conditionals = [(cls.is_chat_model, cls.CHAT_REFINE_TABLE_CONTEXT_PROMPT)]
        
        
        if cls.JP_SIMPLE_INPUT_TMPL is not None:
            cls.JP_SIMPLE_INPUT_PROMPT = PromptTemplate(
                cls.JP_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
            )
        
        if cls.JP_TEXT_QA_PROMPT_TMPL is not None:
            cls.JP_TEXT_QA_PROMPT = PromptTemplate(
                cls.JP_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
            )
        
        if cls.JP_TREE_SUMMARIZE_TMPL is not None:
            cls.JP_TREE_SUMMARIZE_PROMPT = PromptTemplate(
                cls.JP_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
            )
        
        if cls.JP_REFINE_PROMPT_TMPL is not None:
            cls.JP_REFINE_PROMPT = PromptTemplate(
                cls.JP_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
            )
        
        if cls.JP_REFINE_TABLE_CONTEXT_TMPL is not None:
            cls.JP_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
                cls.JP_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
            )
        
        if cls.JP_EVAL_TEMPLATE_TMPL is not None:
            cls.JP_EVAL_TEMPLATE = PromptTemplate(
                cls.JP_EVAL_TEMPLATE_TMPL, prompt_type=PromptType.SIMPLE_INPUT
            )
        
        if cls.JP_REFINE_TEMPLATE_TMPL is not None:
            cls.JP_REFINE_TEMPLATE = PromptTemplate(
                cls.JP_REFINE_TEMPLATE_TMPL, prompt_type=PromptType.REFINE
            )
        
        if cls.JP_SUMMARY_PROMPT_TMPL is not None:
            cls.JP_SUMMARY_PROMPT = PromptTemplate(
                cls.JP_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
            )
        
        if cls.JP_INSERT_PROMPT_TMPL is not None:
            cls.JP_INSERT_PROMPT = PromptTemplate(
                cls.JP_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
            )
        
        if cls.JP_KEYWORD_EXTRACT_TEMPLATE_TMPL is not None:
            cls.JP_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
                cls.JP_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
            )
        
        if cls.JP_TEXT_QA_PROMPT is not None:
            cls.JP_TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
                default_template=cls.JP_TEXT_QA_PROMPT,
                conditionals=cls.default_text_qa_conditionals,
            )
        
        if cls.JP_TREE_SUMMARIZE_PROMPT is not None:
            cls.JP_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
                default_template=cls.JP_TREE_SUMMARIZE_PROMPT,
                conditionals=cls.default_tree_summarize_conditionals,
            )
        
        if cls.JP_REFINE_PROMPT is not None:
            cls.JP_REFINE_PROMPT_SEL = SelectorPromptTemplate(
                default_template=cls.JP_REFINE_PROMPT,
                conditionals=cls.default_refine_conditionals,
            )
        
        if cls.JP_REFINE_TABLE_CONTEXT_PROMPT is not None:
            cls.JP_REFINE_TABLE_CONTEXT_PROMPT_SEL = SelectorPromptTemplate(
                default_template=cls.JP_REFINE_TABLE_CONTEXT_PROMPT,
                conditionals=cls.default_refine_table_conditionals,
            )
    
    @classmethod
    def get_templates_info(cls) -> dict:
        """現在のテンプレート設定情報を取得（テスト記録用）"""
        if not cls._templates_loaded and cls._config_manager:
            cls._load_templates()
        
        return {
            "templates_loaded_from_yaml": cls._templates_loaded,
            "simple_input": cls.JP_SIMPLE_INPUT_TMPL,
            "text_qa": cls.JP_TEXT_QA_PROMPT_TMPL,
            "tree_summarize": cls.JP_TREE_SUMMARIZE_TMPL,
            "refine": cls.JP_REFINE_PROMPT_TMPL,
            "refine_table_context": cls.JP_REFINE_TABLE_CONTEXT_TMPL,
            "single_select": cls.JP_SINGLE_SELECT_PROMPT_TMPL,
            "multi_select": cls.JP_MULTI_SELECT_PROMPT_TMPL,
            "extractor": {
                "title_node": cls.JP_TITLE_NODE_TMPL,
                "title_combine": cls.JP_TITLE_COMBINE_TMPL,
                "summary_extract": cls.JP_SUMMARY_EXTRACT_TMPL,
                "question_gen": cls.JP_QUESTION_GEN_TMPL,
                "keyword_extract": cls.JP_KEYWORD_EXTRACT_TEMPLATE_TMPL,
            },
            "evaluation": {
                "question_generation": cls.JP_QUESTION_GENERATION_PROMPT,
                "question_gen_query": cls.JP_QUESTION_GEN_QUERY,
                "eval_template": cls.JP_EVAL_TEMPLATE_TMPL,
                "refine_template": cls.JP_REFINE_TEMPLATE_TMPL,
                "summary_query": cls.JP_SUMMARY_QUERY,
                "summary_prompt": cls.JP_SUMMARY_PROMPT_TMPL,
                "insert_prompt": cls.JP_INSERT_PROMPT_TMPL,
            }
        }
    
    @staticmethod
    def is_chat_model() -> bool:
        return Settings.llm.metadata.is_chat_model


TemplatePromptSettings = _TemplatePromptSettings()