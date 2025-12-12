import logging
from typing import List, Optional, Dict, Any
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

from .models import ChunkType

logger = logging.getLogger(__name__)


class HybridConversationRetriever(BaseRetriever):
    """概要と会話の2段階検索を行うレトリーバー"""
    
    def __init__(
        self,
        summary_index: BaseIndex,
        conversation_index: BaseIndex,
        similarity_top_k: int = 5,
        enable_two_stage: bool = True,
    ):
        """
        Args:
            summary_index: 概要・トピック用のインデックス
            conversation_index: 会話詳細用のインデックス
            similarity_top_k: 取得する類似ノード数
            enable_two_stage: 2段階検索を有効にするか
        """
        self._summary_index = summary_index
        self._conversation_index = conversation_index
        self._similarity_top_k = similarity_top_k
        self._enable_two_stage = enable_two_stage
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """検索実行"""
        if not self._enable_two_stage:
            # シンプルな検索: 両方から取得してマージ
            return self._simple_retrieve(query_bundle)
        else:
            # 2段階検索: 概要で絞り込み→会話詳細
            return self._two_stage_retrieve(query_bundle)
    
    def _simple_retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """シンプルな検索: 両インデックスから結果を取得"""
        summary_retriever = self._summary_index.as_retriever(
            similarity_top_k=self._similarity_top_k
        )
        conversation_retriever = self._conversation_index.as_retriever(
            similarity_top_k=self._similarity_top_k
        )
        
        summary_nodes = summary_retriever.retrieve(query_bundle)
        conversation_nodes = conversation_retriever.retrieve(query_bundle)
        
        # スコアでソートしてマージ
        all_nodes = summary_nodes + conversation_nodes
        all_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        
        return all_nodes[:self._similarity_top_k]
    
    def _two_stage_retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """2段階検索: 概要で関連セッション特定→会話詳細検索"""
        # Stage 1: 概要から関連セッションを特定
        summary_retriever = self._summary_index.as_retriever(
            similarity_top_k=self._similarity_top_k
        )
        summary_nodes = summary_retriever.retrieve(query_bundle)
        
        if not summary_nodes:
            return []
        
        # 関連セッションのUIDを抽出
        relevant_session_uids = list(set(
            node.node.metadata.get("session_uid")
            for node in summary_nodes
            if node.node.metadata.get("session_uid")
        ))
        
        logger.info(f"Stage 1: {len(relevant_session_uids)}個の関連セッションを特定")
        
        # Stage 2: 特定されたセッションの会話詳細を検索
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="session_uid",
                    value=relevant_session_uids,
                    operator=FilterOperator.IN
                )
            ]
        )
        
        conversation_retriever = self._conversation_index.as_retriever(
            similarity_top_k=self._similarity_top_k * 2,
            filters=filters
        )
        conversation_nodes = conversation_retriever.retrieve(query_bundle)
        
        # 概要ノードと会話ノードを結合（概要を優先的に含める）
        result_nodes = summary_nodes[:2] + conversation_nodes
        result_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        
        return result_nodes[:self._similarity_top_k]


class ChunkTypeFilteredRetriever(BaseRetriever):
    """チャンク種別でフィルタリングするレトリーバー"""
    
    def __init__(
        self,
        index: BaseIndex,
        chunk_types: List[ChunkType],
        similarity_top_k: int = 5,
        additional_filters: Optional[MetadataFilters] = None,
    ):
        """
        Args:
            index: 検索対象のインデックス
            chunk_types: 検索対象のチャンク種別リスト
            similarity_top_k: 取得する類似ノード数
            additional_filters: 追加のメタデータフィルタ
        """
        self._index = index
        self._chunk_types = chunk_types
        self._similarity_top_k = similarity_top_k
        self._additional_filters = additional_filters
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """検索実行"""
        # チャンク種別フィルタを作成
        chunk_type_values = [ct.value for ct in self._chunk_types]
        
        filters_list = [
            MetadataFilter(
                key="chunk_type",
                value=chunk_type_values,
                operator=FilterOperator.IN
            )
        ]
        
        # 追加フィルタがあれば結合
        if self._additional_filters:
            filters_list.extend(self._additional_filters.filters)
        
        filters = MetadataFilters(filters=filters_list)
        
        retriever = self._index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            filters=filters
        )
        
        return retriever.retrieve(query_bundle)

