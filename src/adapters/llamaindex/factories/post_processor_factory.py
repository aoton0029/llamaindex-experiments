import logging
from typing import List, Optional, Any, Dict
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
    FixedRecencyPostprocessor,
    PrevNextNodePostprocessor,
    MetadataReplacementPostProcessor,
    EmbeddingRecencyPostprocessor,
    PIINodePostprocessor    
)

logger = logging.getLogger(__name__)


class PostProcessorFactory:
    """ポストプロセッサーファクトリー"""
    @staticmethod
    def create(type_name: str, **kwargs) -> Any:
        processors = {
            "similarity": PostProcessorFactory.create_similarity_postprocessor,
            "keyword": PostProcessorFactory.create_keyword_postprocessor,
            "recency": PostProcessorFactory.create_recency_postprocessor,
        }
        if type_name not in processors:
            logger.error(f"不明なポストプロセッサータイプ: {type_name}")
            raise ValueError(f"Unknown post-processor type: {type_name}")
        return processors[type_name](**kwargs)
    
    @staticmethod
    def create_similarity_postprocessor(similarity_cutoff: float = 0.7):
        """類似度カットオフポストプロセッサーを作成"""
        try:
            processor = SimilarityPostprocessor(
                similarity_cutoff=similarity_cutoff
            )
            logger.info(f"SimilarityPostprocessorを作成: cutoff={similarity_cutoff}")
            return processor
        except Exception as e:
            logger.error(f"SimilarityPostprocessor作成エラー: {e}")
            raise
    
    @staticmethod
    def create_keyword_postprocessor(
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        lang: str = "ja"
    ):
        """キーワードフィルタポストプロセッサーを作成"""
        try:
            processor = KeywordNodePostprocessor(
                required_keywords=required_keywords or [],
                exclude_keywords=exclude_keywords or [],
                lang=lang
            )
            logger.info(f"KeywordNodePostprocessorを作成")
            return processor
        except Exception as e:
            logger.error(f"KeywordNodePostprocessor作成エラー: {e}")
            raise
    
    @staticmethod
    def create_recency_postprocessor(
        date_key: str = "date",
        top_k: int = 1,
    ):
        """最新性ポストプロセッサーを作成"""
        try:
            processor = FixedRecencyPostprocessor(
                date_key=date_key,
                top_k=top_k,
            )
            logger.info(f"FixedRecencyPostprocessorを作成: top_k={top_k}")
            return processor
        except Exception as e:
            logger.error(f"FixedRecencyPostprocessor作成エラー: {e}")
            raise
    
    
