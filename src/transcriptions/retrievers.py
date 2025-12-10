# filepath: d:\開発\noto\llamaindex-experiments-main\llamaindex-experiments-main\src\transcription\retrievers.py
import logging
from typing import List, Optional, Dict, Any
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.indices.base import BaseIndex
from factories import (
    RetrieverFactory,
    ResponseSynthesizerFactory,
    TemplatePromptSettings,
    DomainLLMSettings,
)

logger = logging.getLogger(__name__)


class ConversationRetriever(BaseRetriever):
    """
    会話情報用カスタムRetriever
    
    機能:
    - ベクトル類似度検索
    - メタデータフィルタ検索（会社名、営業担当者名など）
    """
    
    def __init__(
        self,
        index,
        similarity_top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Args:
            index: VectorStoreIndex
            similarity_top_k: 取得する候補数
            filters: メタデータフィルタ（例: {"会社名": "ABC商事株式会社"}）
        """
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._filters = filters
        super().__init__(**kwargs)
        
        logger.info(f"ConversationRetriever初期化: top_k={similarity_top_k}, filters={filters}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        検索実行
        
        Args:
            query_bundle: クエリ情報
            
        Returns:
            検索結果のNodeリスト
        """
        # メタデータフィルタの構築
        metadata_filters = None
        if self._filters:
            filter_list = []
            for key, value in self._filters.items():
                filter_list.append(
                    MetadataFilter(key=key, value=value, operator=FilterOperator.EQ)
                )
            metadata_filters = MetadataFilters(filters=filter_list)
        
        # Retrieverを作成して検索
        retriever = self._index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            filters=metadata_filters
        )
        
        nodes = retriever.retrieve(query_bundle)
        
        logger.info(f"検索完了: クエリ='{query_bundle.query_str}', 結果={len(nodes)}件")
        
        return nodes


class MultiQueryRetriever(BaseRetriever):
    """
    複数クエリを実行して結果を統合するRetriever
    要約優先検索などに使用
    """
    
    def __init__(
        self,
        index,
        similarity_top_k: int = 5,
        query_priority: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            index: VectorStoreIndex
            similarity_top_k: 各クエリで取得する候補数
            query_priority: 検索優先フィールド（例: ["要約_全体概要", "要約_決定事項"]）
        """
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._query_priority = query_priority or []
        super().__init__(**kwargs)
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        優先フィールドを考慮した検索
        """
        # 基本検索
        retriever = self._index.as_retriever(similarity_top_k=self._similarity_top_k)
        nodes = retriever.retrieve(query_bundle)
        
        # 要約フィールドを持つノードの優先度を上げる
        if self._query_priority:
            for node in nodes:
                for priority_field in self._query_priority:
                    if node.node.metadata.get(priority_field):
                        node.score *= 1.2  # スコアを20%増加
        
        # スコア順にソート
        nodes.sort(key=lambda x: x.score, reverse=True)
        
        return nodes[:self._similarity_top_k]

