# filepath: d:\開発\noto\llamaindex-experiments-main\llamaindex-experiments-main\src\transcription\parsers.py
import logging
from typing import List, Optional
from abc import ABC, abstractmethod
from llama_index.core.schema import Document, BaseNode, TextNode
from models import ConversationSession, ConversationChunkMetadata, Utterance, Topic
from factories import (
    ChunkerFactory,
)
logger = logging.getLogger(__name__)


class ConversationParser:
    """
    会話ドキュメントをチャンクに分割するパーサー
    
    チャンキング戦略:
    - チャンクサイズ: 512トークン
    - オーバーラップ: 128トークン
    - 発話の途中で分割しない（paragraph_separator="\n"）
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Args:
            chunk_size: チャンクサイズ（トークン数）
            chunk_overlap: オーバーラップサイズ（トークン数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # SentenceSplitterを使用（発話単位で分割）
        self.parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n",  # 発話ごとに改行されているため
        )
        
        logger.info(f"ConversationParser初期化: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def parse_documents(self, documents: List[Document]) -> List:
        """
        複数のDocumentをチャンクに分割
        
        Args:
            documents: LlamaIndex Documentのリスト
            
        Returns:
            Nodeのリスト（各チャンクに親Documentのmetadataを継承）
        """
        all_nodes = []
        
        for doc in documents:
            nodes = self.parser.get_nodes_from_documents([doc])
            
            # 各ノードにチャンク情報を追加
            for i, node in enumerate(nodes):
                node.metadata["chunk_index"] = i
                node.metadata["total_chunks"] = len(nodes)
            
            all_nodes.extend(nodes)
        
        logger.info(f"チャンキング完了: {len(documents)}ドキュメント → {len(all_nodes)}チャンク")
        return all_nodes

