import logging
from typing import List, Optional
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import BaseExtractor
from llama_index.core.node_parser import NodeParser

from .chunking_factory import ChunkerFactory
from .extractor_factory import ExtractorFactory

logger = logging.getLogger(__name__)


class TransformationsFactory:
    """
    """
    def __init__(self):
        self.transformations = []
    
    def add_chunker(self, chunker_type: str = "sentance", **kwargs):
        try:
            parser = ChunkerFactory.create(chunker_type, **kwargs)
            self.transformations.append(parser)
            logger.info("")
            return self
        except Exception as e:
            logger.error(f"transformations add chunker error: {e}")
            raise
    
    def add_extractor(self, extractor_type: str, **kwargs):
        try:
            extractor = ExtractorFactory.create_extractor(extractor_type, **kwargs)
            self.transformations.append(extractor)
            logger.info("")
            return self
        except Exception as e:
            logger.error(f"transformations add extractor error: {e}")
            raise
    
    def add_embed_model(self, embed_model = None):
        try:
            if embed_model is None:
                embed_model = Settings.embed_model
            self.transformations.append(embed_model)
            logger.info("")
            return self
        except Exception as e:
            logger.error(f"transformations embed model error: {e}")
            raise
        
    def build(self):
        if not self.transformations:
            logger.warning(f"transformations is empty")
        logger.info(f"build {len(self.transformations)} transformations")
        return self.transformations
    
    def apply_to_settings(self):
        Settings.transformations = self.build()
        logger.info("apply transformations to Settings")
        return Settings.transformations
    
    def reset(self):
        self.transformations = []
        logger.info("reset transformations")
        
    @classmethod
    def create_with_metadata_extraction(cls,
        chunker_type: str,
        chunk_size: int,
        chunk_overlap: int,
        extract_title: bool=False,
        extract_summary: bool=False,
        extract_keywords: bool=False,
        extract_questions: bool=False
    ):
        factory = cls()
        factory.add_chunker(chunker_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if extract_title:
            factory.add_extractor("title")
        if extract_summary:
            factory.add_extractor("summary")
        if extract_keywords:
            factory.add_extractor("keyword")
        if extract_questions:
            factory.add_extractor("questions_answered")
        factory.add_embed_model()
        return factory.build()
        
    
