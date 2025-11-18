from typing import List
import logging
from llama_index.core.schema import Document, BaseNode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from .template_prompts import *

logger = logging.getLogger(__name__)

class ExtractorFactory:
    @staticmethod
    def create_extractor(extractor_type: str, **kwargs):
        if extractor_type == "title":
            return ExtractorFactory.create_title_extractor(**kwargs)
        elif extractor_type == "keyword":
            return ExtractorFactory.create_keyword_extractor(**kwargs)
        elif extractor_type == "summary":
            return ExtractorFactory.create_summary_extractor(**kwargs)
        elif extractor_type == "questions_answered":
            return ExtractorFactory.create_questions_answered_extractor(**kwargs)
        else:
            raise ValueError(f"未知のエクストラクタータイプ: {extractor_type}")
    
    @staticmethod
    def create_title_extractor(nodes: int = 5):
        try:
            extractor = TitleExtractor(
                nodes = nodes,
                default_title_node_template = DEFAULT_TITLE_NODE_TEMPLATE,
                default_title_combine_template = DEFAULT_TITLE_COMBINE_TEMPLATE
            )
            logger.info("TitleExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"TitleExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def create_summary_extractor():
        try:
            extractor = SummaryExtractor(
                prompt_template=DEFAULT_SUMMARY_EXTRACT_TEMPLATE
            )
            logger.info("SummaryExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"SummaryExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def create_keyword_extractor():
        try:
            extractor = KeywordExtractor(
                prompt_template=DEFAULT_KEYWORD_EXTRACT_TEMPLATE
            )
            logger.info("KeywordExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"KeywordExtractor作成エラー: {e}")
            raise
    
    @staticmethod
    def create_questions_answered_extractor(questions: int = 5):
        try:
            extractor = QuestionsAnsweredExtractor(
                questions=questions,
                prompt_template=DEFAULT_QUESTION_GEN_TMPL
            )
            logger.info("QuestionsAnsweredExtractorを作成")
            return extractor
        except Exception as e:
            logger.error(f"QuestionsAnsweredExtractor作成エラー: {e}")
            raise