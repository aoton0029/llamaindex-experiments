import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import (
    NodeParser,
    SimpleNodeParser, 
    SentenceSplitter,   
    SemanticSplitterNodeParser,
    LangchainNodeParser,
    MarkdownNodeParser,
    HierarchicalNodeParser,
)
from llama_index.core.text_splitter import TokenTextSplitter

logger = logging.getLogger(__name__)



class ChunkerFactory:
    @staticmethod
    def create(chunker_type: str, **kwargs) -> NodeParser:
        chunk_types = {
            "simple": ChunkerFactory._create_simple_node_parser,
            "sentence": ChunkerFactory._create_sentence_splitter,
            "token": ChunkerFactory._create_token_text_splitter,
            "semantic": ChunkerFactory._create_semantic_splitter,
            "hierarchial": ChunkerFactory._create_hieralchial_node_parser,
        }
        
        if chunker_type in chunk_types:
            return chunk_types[chunker_type](**kwargs)
        else:
            raise ValueError(f"未知のチャンカータイプ: {chunker_type}")
    
    @staticmethod
    def _create_simple_node_parser(
        chunk_size: int = 512, 
        chunk_overlap: int = 20, 
        paragraph_separator: str = "\n\n\n", 
        secondary_chunking_regex: str = ""
    ) -> SimpleNodeParser:
        try:
            return SimpleNodeParser.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                paragraph_separator=paragraph_separator,
                secondary_chunking_regex=secondary_chunking_regex
            )
        except Exception as e:
            logger.error(f"SimpleNodeParser作成エラー: {e}")
            raise
        
        
    @staticmethod
    def _create_sentence_splitter(
        chunk_size: int = 512, 
        chunk_overlap: int = 20,
        separator: str = "。",
        paragraph_separator: str = "\n\n\n",
        secondary_chunking_regex: str = ""
    ) -> SentenceSplitter:
        """SentenceSplitterを作成"""
        try:
            return SentenceSplitter.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator,
                paragraph_separator=paragraph_separator,
                secondary_chunking_regex=secondary_chunking_regex,
            )
        except Exception as e:
            logger.error(f"SentenceSplitter作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_token_text_splitter(
        chunk_size: int = 512, 
        chunk_overlap: int = 20,
        separator: str = "。",
        backup_separators: Optional[List[str]] = None
    ) -> TokenTextSplitter:        
        """TokenTextSplitterを使用したNodeParserを作成"""
        try:
            return TokenTextSplitter.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator,
                backup_separators=backup_separators
            )
        except Exception as e:
            logger.error(f"TokenTextSplitter作成エラー: {e}")
            raise

    @staticmethod
    def _create_semantic_splitter(
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95
    ) -> SemanticSplitterNodeParser:
        """SemanticSplitterNodeParserを作成"""
        try:
            return SemanticSplitterNodeParser.from_defaults(
                buffer_size=buffer_size,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold
            )
        except Exception as e:
            logger.error(f"SemanticSplitterNodeParser作成エラー: {e}")
            raise
    
    @staticmethod
    def _create_hieralchial_node_parser(
        chunk_sizes: List[int],
        chunk_overlap: int,
    ) -> HierarchicalNodeParser:
        try:
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=chunk_sizes,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            logger.error(f"HierarchicalNodeParser作成エラー: {e}")
            raise


