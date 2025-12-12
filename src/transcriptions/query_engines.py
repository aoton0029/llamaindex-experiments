import logging
from typing import List, Optional, Dict, Any
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex

from .retrievers import HybridConversationRetriever, ChunkTypeFilteredRetriever
from .models import ChunkType

logger = logging.getLogger(__name__)


class ConversationQueryEngineFactory:
    """会話情報RAG用のクエリエンジンファクトリ"""
    
    # プロンプトテンプレート定義
    QA_PROMPT_TEMPLATE = PromptTemplate(
        """あなたは営業支援AIアシスタントです。
会話の文字起こしデータと概要情報を基に、取引先が何を求めているかを分析してください。

以下の情報を参考にして質問に答えてください:

{context_str}

質問: {query_str}

回答する際は、以下の点に注意してください:
1. 会社名・担当者名・トピックを明記する
2. 具体的な発言内容を引用する
3. 時間情報があれば記載する（[XX.Xs-YY.Ys]）
4. 情報が不足している場合は「情報が見つかりませんでした」と明記する

回答:"""
    )
    
    REFINE_PROMPT_TEMPLATE = PromptTemplate(
        """既存の回答を追加情報で補完してください。

元の質問: {query_str}
既存の回答: {existing_answer}

追加情報:
{context_msg}

追加情報を踏まえて回答を改善してください。情報が重複している場合は統合し、新しい情報があれば追加してください。

改善された回答:"""
    )
    
    @staticmethod
    def create_hybrid_query_engine(
        summary_index: VectorStoreIndex,
        conversation_index: VectorStoreIndex,
        similarity_top_k: int = 5,
        enable_two_stage: bool = True,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> RetrieverQueryEngine:
        """
        2段階検索を行うクエリエンジンを作成
        
        Args:
            summary_index: 概要・トピック用のインデックス
            conversation_index: 会話詳細用のインデックス
            similarity_top_k: 取得する類似ノード数
            enable_two_stage: 2段階検索を有効にするか
            response_mode: レスポンス生成モード
        """
        retriever = HybridConversationRetriever(
            summary_index=summary_index,
            conversation_index=conversation_index,
            similarity_top_k=similarity_top_k,
            enable_two_stage=enable_two_stage,
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            text_qa_template=ConversationQueryEngineFactory.QA_PROMPT_TEMPLATE,
            refine_template=ConversationQueryEngineFactory.REFINE_PROMPT_TEMPLATE,
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
    
    @staticmethod
    def create_summary_only_query_engine(
        index: BaseIndex,
        similarity_top_k: int = 5,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> RetrieverQueryEngine:
        """
        概要・トピックのみを検索するクエリエンジン
        大まかな質問（「どの会社が〇〇に興味があるか」など）に適している
        """
        retriever = ChunkTypeFilteredRetriever(
            index=index,
            chunk_types=[ChunkType.SUMMARY, ChunkType.TOPIC],
            similarity_top_k=similarity_top_k,
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            text_qa_template=ConversationQueryEngineFactory.QA_PROMPT_TEMPLATE,
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
    
    @staticmethod
    def create_conversation_only_query_engine(
        index: BaseIndex,
        similarity_top_k: int = 5,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ) -> RetrieverQueryEngine:
        """
        会話詳細のみを検索するクエリエンジン
        具体的な発言内容を知りたい場合に適している
        """
        retriever = ChunkTypeFilteredRetriever(
            index=index,
            chunk_types=[ChunkType.CONVERSATION],
            similarity_top_k=similarity_top_k,
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            text_qa_template=ConversationQueryEngineFactory.QA_PROMPT_TEMPLATE,
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

