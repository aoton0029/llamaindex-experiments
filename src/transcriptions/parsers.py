# filepath: d:\開発\noto\llamaindex-experiments-main\llamaindex-experiments-main\src\transcription\parsers.py
import logging
from typing import List, Optional
from abc import ABC, abstractmethod
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.node_parser import NodeParser

from models import ConversationSession, ConversationChunkMetadata, Utterance
from src.adapters.llamaindex.factories import ChunkerFactory


logger = logging.getLogger(__name__)


class ConversationParser:
    """
    会話ドキュメントをチャンクに分割するパーサー
    
    チャンキング戦略:
    - チャンクサイズ: 512トークン
    - オーバーラップ: 128トークン
    - 発話の途中で分割しない（paragraph_separator="\n"）
    """
