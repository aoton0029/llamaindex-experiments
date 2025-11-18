import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import (
    NodeParser,
    SimpleNodeParser, 
    SentenceSplitter,   
    SemanticSplitterNodeParser
)
from llama_index.core.text_splitter import TokenTextSplitter


logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    def __init__(self):
        self._parser = None
    
    @abstractmethod
    def _create_parser(self) -> NodeParser:
        """NodeParserを作成"""
        pass
    
    def get_parser(self) -> NodeParser:
        """NodeParserを取得"""
        if self._parser is None:
            self._parser = self._create_parser()
        return self._parser
    
    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        parser = self.get_parser()
        nodes = parser.get_nodes_from_documents(documents)
        
        logger.info(f"{len(documents)}ドキュメントから{len(nodes)}ノードを作成")
        return nodes
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[BaseNode]:
        document = Document(text=text, metadata=metadata or {})
        return self.chunk_documents([document])
    


class SimpleNodeParserChunker(BaseChunker):
    def __init__(self, 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 20, 
                 include_metadata: bool = True, 
                 include_prev_next_rel: bool = False):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel

    def _create_parser(self) -> SimpleNodeParser:
        """SimpleNodeParserを作成"""
        try:
            return SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                include_metadata=self.include_metadata,
                include_prev_next_rel=self.include_prev_next_rel
            )
        except Exception as e:
            logger.error(f"SimpleNodeParser作成エラー: {e}")
            raise


class SentenceSplitterChunker(BaseChunker):
    def __init__(self, 
                chunk_size: int = 512, 
                chunk_overlap: int = 20,
                separator: str = " ",
                paragraph_separator: str = "\n\n\n"):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator  
        self.paragraph_separator = paragraph_separator
    
    def _create_parser(self) -> SentenceSplitter:
        """SentenceSplitterを作成"""
        try:
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                paragraph_separator=self.paragraph_separator,
                separator=self.separator
            )
        except Exception as e:
            logger.error(f"SentenceSplitter作成エラー: {e}")
            raise


class TokenBasedChunker(BaseChunker):
    def __init__(self, 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 20,
                 separator: str = " "):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def _create_parser(self) -> NodeParser:
        """TokenTextSplitterを使用したNodeParserを作成"""
        try:
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=self.separator
            )
        except Exception as e:
            logger.error(f"TokenBasedChunker作成エラー: {e}")
            raise


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        buffer_size: int = 1,
        include_metadata: bool = True,
        include_prev_next_rel: bool = False,
        breakpoint_percentile_threshold: int = 95
    ):
        super().__init__()
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
    
    def _create_parser(self) -> NodeParser:
        """SemanticSplitterNodeParserを作成"""
        try:
            return SemanticSplitterNodeParser(
                buffer_size=self.buffer_size,
                include_metadata=self.include_metadata,
                include_prev_next_rel=self.include_prev_next_rel,
                breakpoint_percentile_threshold=self.breakpoint_percentile_threshold
            )
        except Exception as e:
            logger.error(f"SemanticChunker作成エラー: {e}")
            raise


class ChunkerFactory:
    @staticmethod
    def create(chunker_type: str, **kwargs) -> BaseChunker:
        if chunker_type == "simple":
            return SimpleNodeParserChunker(**kwargs)
        elif chunker_type == "sentence":
            return SentenceSplitterChunker(**kwargs)
        elif chunker_type == "token":
            return TokenBasedChunker(**kwargs)
        elif chunker_type == "semantic":
            return SemanticChunker(**kwargs)
        else:
            raise ValueError(f"未知のチャンカータイプ: {chunker_type}")
        


