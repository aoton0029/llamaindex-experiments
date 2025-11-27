from dataclasses import dataclass
from typing import Any, Optional, Dict
from llama_index.core import Settings
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate

@dataclass
class _TemplatePromptSettings:
    """テンプレートプロンプト設定クラス
    
    デフォルトはNone（LlamaIndexのデフォルトを使用）。
    YAMLファイルから設定を読み込んでオーバーライドできる。
    """
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
                if "jp_text_qa_tmpl" in text_qa:
                    cls.JP_TEXT_QA_PROMPT_TMPL = text_qa["jp_text_qa_tmpl"]
            
            if "tree_summarize" in templates:
                tree_summarize = templates["tree_summarize"]
                if "jp_tree_summarize_tmpl" in tree_summarize:
                    cls.JP_TREE_SUMMARIZE_TMPL = tree_summarize["jp_tree_summarize_tmpl"]
            
            if "refine" in templates:
                refine = templates["refine"]
                if "jp_refine_tmpl" in refine:
                    cls.JP_REFINE_PROMPT_TMPL = refine["jp_refine_tmpl"]
            
            if "refine_table_context" in templates:
                refine_table = templates["refine_table_context"]
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
                    cls.JP_TITLE_NODE_TEMPLATE = extractor["jp_title_node_template_tmpl"]
                if "jp_title_combine_template_tmpl" in extractor:
                    cls.JP_TITLE_COMBINE_TEMPLATE = extractor["jp_title_combine_template_tmpl"]
                if "jp_summary_extract_template_tmpl" in extractor:
                    cls.JP_SUMMARY_EXTRACT_TEMPLATE = extractor["jp_summary_extract_template_tmpl"]
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
                if "jp_summary_query_tech" in evaluation:
                    cls.JP_SUMMARY_QUERY_TECH = evaluation["jp_summary_query_tech"]
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
        
        # Selector prompts (デフォルトテンプレートがNoneでない場合のみ生成)
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
                "title_node": cls.JP_TITLE_NODE_TEMPLATE,
                "title_combine": cls.JP_TITLE_COMBINE_TEMPLATE,
                "summary_extract": cls.JP_SUMMARY_EXTRACT_TEMPLATE,
                "question_gen": cls.JP_QUESTION_GEN_TMPL,
                "keyword_extract": cls.JP_KEYWORD_EXTRACT_TEMPLATE_TMPL,
            },
            "evaluation": {
                "question_generation": cls.JP_QUESTION_GENERATION_PROMPT,
                "question_gen_query": cls.JP_QUESTION_GEN_QUERY,
                "eval_template": cls.JP_EVAL_TEMPLATE_TMPL,
                "refine_template": cls.JP_REFINE_TEMPLATE_TMPL,
                "summary_query": cls.JP_SUMMARY_QUERY,
                "summary_query_tech": cls.JP_SUMMARY_QUERY_TECH,
                "summary_prompt": cls.JP_SUMMARY_PROMPT_TMPL,
                "insert_prompt": cls.JP_INSERT_PROMPT_TMPL,
            }
        }
    
    @staticmethod
    def is_chat_model() -> bool:
        return Settings.llm.metadata.is_chat_model

    # デフォルト値を None に設定（LlamaIndexのデフォルトを使用）
    # YAMLで設定されている場合のみオーバーライドされる
    
    # Simple Input
    JP_SIMPLE_INPUT_TMPL = None
    JP_SIMPLE_INPUT_PROMPT = None

    # Text QA
    TEXT_QA_SYSTEM_PROMPT = ChatMessage(
        content=(
            "あなたは世界中で信頼されている専門のQ&Aシステムです。\n"
            "常に提供されたコンテキスト情報のみを用いて質問に回答し、事前の知識は使用しないでください。\n"
            "従うべきルール:\n"
            "1. 回答内で与えられたコンテキストを直接参照しないでください。\n"
            "2. 「コンテキストに基づいて...」や「コンテキスト情報...」のような表現を避けてください。"
        ),
        role=MessageRole.SYSTEM,
    )

    TEXT_QA_PROMPT_TMPL_MSGS = [
        TEXT_QA_SYSTEM_PROMPT,
        ChatMessage(
            content=(
                "以下にコンテキスト情報を示します。\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "コンテキスト情報のみを用いて質問に回答してください（事前知識は使用しないでください）。\n"
                "質問: {query_str}\n"
                "回答: "
            ),
            role=MessageRole.USER,
        ),
    ]

    CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

    JP_TEXT_QA_PROMPT_TMPL = None
    JP_TEXT_QA_PROMPT = None

    default_text_qa_conditionals = [(is_chat_model, CHAT_TEXT_QA_PROMPT)]
    JP_TEXT_QA_PROMPT_SEL = None

    # Tree Summarize
    TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
        TEXT_QA_SYSTEM_PROMPT,
        ChatMessage(
            content=(
                "複数のソースからのコンテキスト情報を以下に示します。\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "複数の情報を踏まえて（事前知識は使用せずに）質問に回答してください。\n"
                "質問: {query_str}\n"
                "回答: "
            ),
            role=MessageRole.USER,
        ),
    ]

    CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
        message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
    )

    JP_TREE_SUMMARIZE_TMPL = None
    JP_TREE_SUMMARIZE_PROMPT = None
    default_tree_summarize_conditionals = [(is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT)]
    JP_TREE_SUMMARIZE_PROMPT_SEL = None

    # Refine Prompt
    CHAT_REFINE_PROMPT_TMPL_MSGS = [
        ChatMessage(
            content=(
                "あなたは既存の回答を洗練する際、厳密に次の2つのモードで動作する専門のQ&Aシステムです：\n"
                "1. 新しいコンテキストを用いて元の回答を**書き直す**。\n"
                "2. 新しいコンテキストが有用でない場合は元の回答を**繰り返す**。\n"
                "回答内で元の回答やコンテキストを直接参照しないでください。\n"
                "迷ったら元の回答を繰り返してください。\n"
                "新しいコンテキスト: {context_msg}\n"
                "質問: {query_str}\n"
                "元の回答: {existing_answer}\n"
                "新しい回答: "
            ),
            role=MessageRole.USER,
        )
    ]

    CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)

    JP_REFINE_PROMPT_TMPL = None
    JP_REFINE_PROMPT = None
    default_refine_conditionals = [(is_chat_model, CHAT_REFINE_PROMPT)]
    JP_REFINE_PROMPT_SEL = None

    # Refine Table Context
    CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
        ChatMessage(content="{query_str}", role=MessageRole.USER),
        ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
        ChatMessage(
            content=(
                 "以下にテーブルのスキーマを示します。\n"
                "---------------------\n"
                "{schema}\n"
                "---------------------\n"
                "さらにコンテキスト情報を以下に示します。{context_msg}\n"
                "---------------------\n"
                "テーブルのスキーマとコンテキスト情報を用いて元の回答を改善してください。"
                "コンテキストが有用でない場合は元の回答を返してください。"
            ),
            role=MessageRole.USER,
        ),
    ]
    CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
        message_templates=CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
    )
    
    JP_REFINE_TABLE_CONTEXT_TMPL = None
    JP_REFINE_TABLE_CONTEXT_PROMPT = None
    default_refine_table_conditionals = [(is_chat_model, CHAT_REFINE_TABLE_CONTEXT_PROMPT)]
    JP_REFINE_TABLE_CONTEXT_PROMPT_SEL = None

    # Selection
    JP_SINGLE_SELECT_PROMPT_TMPL = None
    JP_MULTI_SELECT_PROMPT_TMPL = None

    # Extractor
    JP_TITLE_NODE_TEMPLATE = None
    JP_TITLE_COMBINE_TEMPLATE = None
    JP_SUMMARY_EXTRACT_TEMPLATE = None
    JP_QUESTION_GEN_TMPL = None

    # Evaluation
    JP_QUESTION_GENERATION_PROMPT = None
    JP_QUESTION_GEN_QUERY = None
    JP_EVAL_TEMPLATE_TMPL = None
    JP_EVAL_TEMPLATE = None
    JP_REFINE_TEMPLATE_TMPL = None
    JP_REFINE_TEMPLATE = None
    JP_SUMMARY_QUERY = None
    JP_SUMMARY_QUERY_TECH = None
    JP_SUMMARY_PROMPT_TMPL = None
    JP_SUMMARY_PROMPT = None
    JP_INSERT_PROMPT_TMPL = None
    JP_INSERT_PROMPT = None
    JP_KEYWORD_EXTRACT_TEMPLATE_TMPL = None
    JP_KEYWORD_EXTRACT_TEMPLATE = None

TemplatePromptSettings = _TemplatePromptSettings()
