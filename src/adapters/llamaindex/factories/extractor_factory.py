import logging
from typing import List, Optional, Sequence
from llama_index.core.schema import Document, BaseNode
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.extractors import (
    BaseExtractor,
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    DocumentContextExtractor,
    PydanticProgramExtractor
)
from .settings_template_prompts import TemplatePromptSettings
from .settings_llm import DomainLLMSettings
from output_parser_factory import PydanticOutputParserJp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydanticモデル定義
class TitleOutput(BaseModel):
    """タイトル抽出の出力モデル"""
    title: str = Field(description="ドキュメントのタイトル（簡潔で内容を表すもの）")


class SummaryOutput(BaseModel):
    """要約抽出の出力モデル"""
    summary: str = Field(description="ドキュメントの要約（200文字程度）")


class KeywordOutput(BaseModel):
    """キーワード抽出の出力モデル"""
    keywords: List[str] = Field(description="ドキュメントの主要キーワードリスト（最大10個）")


# 構造化Extractorクラス
class StructuredTitleExtractor(BaseExtractor):
    """Pydanticモデルを使用した構造化タイトル抽出"""
    
    _program: LLMTextCompletionProgram = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParserJp(output_cls=TitleOutput),
            output_cls=TitleOutput,
            prompt_template_str=TemplatePromptSettings.TITLE_NODE_TMPL,
            verbose=True
        )
    
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[dict]:
        """非同期でタイトルを抽出"""
        metadata_list = []
        
        for node in nodes:
            try:
                result = await self._program.acall(context_str=node.get_content())
                metadata = {"document_title": result.title}
                logger.info(f"タイトル抽出成功: {result.title}")
            except Exception as e:
                logger.error(f"タイトル抽出エラー: {e}")
                metadata = {"document_title": ""}
            metadata_list.append(metadata)

        return metadata_list


class StructuredSummaryExtractor(BaseExtractor):
    """Pydanticモデルを使用した構造化要約抽出"""
    
    _program: LLMTextCompletionProgram = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        self._program = LLMTextCompletionProgram.from_defaults(
            output_cls=SummaryOutput,
            prompt_template_str=TemplatePromptSettings.JP_SUMMARY_EXTRACT_TMPL,
            verbose=True
        )
    
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[dict]:
        """非同期で要約を抽出"""
        metadata_list = []
        
        for node in nodes:
            try:
                result = await self._program.acall(context_str=node.get_content())
                metadata = {"section_summary": result.summary}
                logger.info(f"要約抽出成功: {result.summary[:50]}...")
            except Exception as e:
                logger.error(f"要約抽出エラー: {e}")
                metadata = {"section_summary": ""}
            
            metadata_list.append(metadata)
        
        return metadata_list


class StructuredKeywordExtractor(BaseExtractor):
    """Pydanticモデルを使用した構造化キーワード抽出"""
    
    _program: LLMTextCompletionProgram = PrivateAttr()
    
    def __init__(self, keywords: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._program = LLMTextCompletionProgram.from_defaults(
            output_cls=KeywordOutput,
            prompt_template_str=TemplatePromptSettings.KEYWORD_EXTRACT_TEMPLATE_TMPL.format(keywords=keywords),
            verbose=True
        )

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[dict]:
        """非同期でキーワードを抽出"""
        metadata_list = []
        
        for node in nodes:
            try:
                result = await self._program.acall(context_str=node.get_content())
                metadata = {"excerpt_keywords": ", ".join(result.keywords)}
                logger.info(f"キーワード抽出成功: {result.keywords}")
            except Exception as e:
                logger.error(f"キーワード抽出エラー: {e}")
                metadata = {"excerpt_keywords": ""}
            
            metadata_list.append(metadata)
        
        return metadata_list


class ExtractorFactory:
    @staticmethod
    def create_extractor(extractor_type: str, **kwargs):
        extractors = {
            "title": ExtractorFactory._create_title_extractor,
            "summary": ExtractorFactory._create_summary_extractor,
            "keyword": ExtractorFactory._create_keyword_extractor,
            "questions_answered": ExtractorFactory._create_questions_answered_extractor,
            # "pydantic_program": ExtractorFactory._create_pydantic_program_extractor,
            # "structured_title": ExtractorFactory._create_structured_title_extractor,
            # "structured_summary": ExtractorFactory._create_structured_summary_extractor,
            # "structured_keyword": ExtractorFactory._create_structured_keyword_extractor,
        }
        if extractor_type in extractors:
            return extractors[extractor_type](**kwargs)
        else:
            raise ValueError(f"未知のエクストラクタータイプ: {extractor_type}")
    
    @staticmethod
    def _create_title_extractor(nodes: int = 5):
        """
        
        'document_title'
        """
        try:
            extractor = TitleExtractor(
                llm=DomainLLMSettings.EXTRACTOR_TITLE,
                nodes = nodes,
                node_template = TemplatePromptSettings.TITLE_NODE_TMPL,
                combine_template = TemplatePromptSettings.TITLE_COMBINE_TMPL
            )
            logger.info(f"TitleExtractorを作成 nodes:{nodes}")
            return extractor
        except Exception as e:
            logger.error(f"TitleExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_summary_extractor():
        """
        
        'section_summary'
        """
        try:
            extractor = SummaryExtractor(
                llm=DomainLLMSettings.EXTRACTOR_SUMMARY,
                prompt_template=TemplatePromptSettings.SUMMARY_EXTRACT_TMPL
            )
            logger.info("SummaryExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"SummaryExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_keyword_extractor(keywords: int = 10):
        """
        
        'excerpt_keywords'
        """
        try:
            extractor = KeywordExtractor(
                llm=DomainLLMSettings.EXTRACTOR_KEYWORD,
                prompt_template=TemplatePromptSettings.KEYWORD_EXTRACT_TEMPLATE_TMPL,
                keywords=keywords
            )
            logger.info(f"KeywordExtractorを作成 keywords:{keywords}")
            return extractor
        except Exception as e:
            logger.error(f"KeywordExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_questions_answered_extractor(questions: int = 5):
        try:
            extractor = QuestionsAnsweredExtractor(
                llm=DomainLLMSettings.EXTRACTOR_QA,
                questions=questions,
                prompt_template=TemplatePromptSettings.QUESTION_GEN_TMPL
            )
            logger.info("QuestionsAnsweredExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"QuestionsAnsweredExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_pydantic_program_extractor(program, input_key: str = "input", **kwargs):
        try:
            extractor = PydanticProgramExtractor(
                program=program,
                input_key=input_key,
                **kwargs
            )
            logger.info("PydanticProgramExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"PydanticProgramExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_structured_title_extractor(**kwargs) -> StructuredTitleExtractor:
        try:
            extractor = StructuredTitleExtractor(**kwargs)
            logger.info("StructuredTitleExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"StructuredTitleExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_structured_summary_extractor(**kwargs) -> StructuredSummaryExtractor:
        try:
            extractor = StructuredSummaryExtractor(**kwargs)
            logger.info("StructuredSummaryExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"StructuredSummaryExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_structured_keyword_extractor(**kwargs) -> StructuredKeywordExtractor:
        try:
            extractor = StructuredKeywordExtractor(**kwargs)
            logger.info("StructuredKeywordExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"StructuredKeywordExtractor作成エラー: {e}")
            raise
