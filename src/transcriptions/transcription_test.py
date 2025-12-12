"""
会話情報RAGシステムのテストスクリプト
"""
import os
import logging
import dotenv
from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_index.core import Settings
from loaders import DocumentLoader
from models import ConversationSession, ConversationChunkMetadata, ChunkType
from ..infrastructure.relational_stores.query_executor import QueryExecutor
from ..infrastructure.storage.storage_context_config import *
from ..runners.test_runner_base import TestRunnerBase


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionTestConfig:
    """会話情報RAGシステムの設定クラス"""
    TOKENIZER_MODEL: str = "models/Qwen/Qwen3-32B-AWQ"
    LLM_MODEL: str = "models/Qwen/Qwen3-32B-AWQ"
    EMBEDDING_MODEL: str = "qwen3-embedding:8b"
    EMBEDDING_DIM: int = 8192
    STORAGE_CONTEXT_CONFIG: StorageContextConfig = StorageContextConfig(
        context_name="transcription",  # ドメイン名
        docstore=DocstoreConfig(
            namespace="transcription", 
            collection_name="transcription_docstore"
        ),
        indexstore=IndexStoreConfig(
            namespace="transcription", 
            collection_suffix="transcription_indexstore"
        ),
        vectorstore=VectorStoreConfig(
            collection_name="transcription_vectors",  # ドメイン全体で共有
            dim=EMBEDDING_DIM,
            schema=ConversationChunkMetadata.schema(EMBEDDING_DIM),
            metric_type="COSINE",
            index_type="HNSW",
            additional_params={"M": 16, "efConstruction": 256},
        ),
    )

class TranscriptionTest(TestRunnerBase):
    """会話情報RAGシステムのテストクラス"""
    
    def __init__(self):
        self.config = TranscriptionTestConfig()
        dotenv.load_dotenv()
        super().__init__()
        self._setup_callback()
        

    def _load_data(self, query_executor: QueryExecutor) -> List[ConversationSession]:
        """会話セッションデータの取得"""
        try:
            self.loader = DocumentLoader(query_executor)
            cache_file_path = "conversation_sessions_cache.pkl"
            if os.path.exists(cache_file_path):
                import pickle
                with open(cache_file_path, 'rb') as f:
                    sessions = pickle.load(f)
                self.monitor.log_event("info", f"会話セッションをキャッシュから読み込み完了: {len(sessions)}件")
            else:
                sessions = self.loader.load_data()
                self.monitor.log_event("info", f"会話セッション取得完了: {len(sessions)}件")
                # キャッシュ保存
                with open(cache_file_path, 'wb') as f:
                    import pickle
                    pickle.dump(sessions, f)
                self.monitor.log_event("info", f"会話セッションをキャッシュに保存: {cache_file_path}")
            return sessions
        except Exception as e:
            self.monitor.log_event("error", f"データ取得エラー: {e}")
            raise


    def run_indexing(self):
        """インデクシング処理の実行（会話セッションごと）"""
        try:
            self.monitor.log_event("info", "=== インデクシング開始 ===")
                        
            # 1. LLMの設定
            Settings.llm = self._setup_llm(self.config.LLM_DOMAIN)
            # 2. Embeddingの設定
            Settings.embed_model = self._setup_embedding(
                model_name=self.config.EMBEDDING_MODEL,
                dim=self.config.EMBEDDING_DIM
            )
            # 3. tokenizer設定
            Settings.tokenizer = self._setup_tokenizer(self.config.TOKENIZER_MODEL)
            # 4. PromptHelper設定
            Settings.prompt_helper = self._setup_prompt_helper()
            # 5. データベース/StorageContext設定
            self._setup_storage_context_manager(
                vector_store_manager=self._setup_vector_store_manager(),
                document_store_manager=self._setup_document_store_manager(),
                index_store_manager=self._setup_index_store_manager()
            )
            # 6. ドメイン共通のStorageContext取得（全セッションで共有）
            storage_context = self._create_storage_context(self.config.STORAGE_CONTEXT_CONFIG)
            self.monitor.log_event("info", f"StorageContext作成: {self.config.STORAGE_CONTEXT_CONFIG.context_name}")
            
            # 7. 会話セッションデータ取得
            db_manager = self._setup_relational_store_manager()
            query_executor = QueryExecutor(db_manager)
            sessions = self._load_data(query_executor)
            if not sessions:
                self.monitor.log_event("warning", "会話セッションが0件です")
                return None
            
            # 8. ConversationSession → Document変換とインデックス作成（セッションごと）
            from .parsers import ConversationDocumentConverter, ConversationNodeParser
            from llama_index.core import VectorStoreIndex
            
            # 9. チャンキング設定（ConversationNodeParserを使用）
            node_parser = ConversationNodeParser(
                chunk_size=512,
                chunk_overlap=50
            )
            
            # 10. セッションごとにインデックス作成（ドメイン共通のStorageContextを使用）
            session_indices = {}
            summary_docs_count = 0
            conversation_docs_count = 0
            
            for idx, session in enumerate(sessions, 1):
                self.monitor.log_event("info", f"セッション {idx}/{len(sessions)}: {session.uid} - {session.company_name}")
                
                # Document変換
                docs = ConversationDocumentConverter.session_to_documents(session)
                
                if not docs:
                    self.monitor.log_event("warning", f"  セッション {session.uid} のドキュメントが0件")
                    continue
                
                # 概要・トピック用のドキュメント
                summary_docs = [doc for doc in docs 
                               if doc.metadata.get("chunk_type") in [ChunkType.SUMMARY.value, ChunkType.TOPIC.value]]
                # 会話詳細用のドキュメント
                conversation_docs = [doc for doc in docs 
                                    if doc.metadata.get("chunk_type") == ChunkType.CONVERSATION.value]
                
                summary_docs_count += len(summary_docs)
                conversation_docs_count += len(conversation_docs)
                
                # 概要インデックス作成（ドメイン共通のStorageContextを使用）
                if summary_docs:
                    summary_index = VectorStoreIndex.from_documents(
                        summary_docs,
                        storage_context=storage_context,
                        transformations=[node_parser],
                        show_progress=False
                    )
                    self.monitor.log_event("info", f"  概要インデックス作成: ({len(summary_docs)}件)")
                
                # 会話インデックス作成（同じStorageContextを使用）
                if conversation_docs:
                    conversation_index = VectorStoreIndex.from_documents(
                        conversation_docs,
                        storage_context=storage_context,
                        transformations=[node_parser],
                        show_progress=False
                    )
                    self.monitor.log_event("info", f"  会話インデックス作成: ({len(conversation_docs)}件)")
            
                                    
            self.monitor.log_event("info", f"全インデックス作成完了: {len(session_indices)}個")
            self.monitor.log_event("info", f"  概要ドキュメント合計: {summary_docs_count}件")
            self.monitor.log_event("info", f"  会話ドキュメント合計: {conversation_docs_count}件")
            self.monitor.log_event("info", "インデックスはStorageContextに保存されました")
                        
        except Exception as e:
            self.monitor.log_event("error", f"インデクシングエラー: {e}")
            raise

    def run_query(self, queries: Optional[List[Dict[str, Any]]] = None):
        """クエリ処理の実行"""
        try:
            self.monitor.log_event("info", "=== クエリ処理開始 ===")
                        
            # 1. LLMの設定
            if not Settings.llm:
                Settings.llm = self._setup_llm(self.config.LLM_DOMAIN)
            
            # 2. Embeddingの設定
            if not Settings.embed_model:
                Settings.embed_model = self._setup_embedding(
                    model_name=self.config.EMBEDDING_MODEL,
                    dim=self.config.EMBEDDING_DIM
                )
            
            # 3. StorageContext設定
            self._setup_storage_context_manager(
                vector_store_manager=self._setup_vector_store_manager(),
                document_store_manager=self._setup_document_store_manager(),
                index_store_manager=self._setup_index_store_manager()
            )
            
            # 4. インデックスの復元
            try:
                # StorageContextを取得（ドメイン共通）
                storage_context = self._create_storage_context(self.config.STORAGE_CONTEXT_CONFIG)
                indices = self.storage_context_manager.load_indices(self.config.STORAGE_CONTEXT_CONFIG.context_name)
                if not indices:
                    self.monitor.log_event("error", "復元可能なインデックスが見つかりません。先にrun_indexing()を実行してください")
                    return
            except Exception as e:
                self.monitor.log_event("error", f"インデックス復元エラー: {e}")
                self.monitor.log_event("info", "先にrun_indexing()を実行してください")
                return
        
            # デフォルトクエリ
            if queries is None:
                queries = [
                    {
                        "query": "ABC商事はどのようなシステムを求めていますか？",
                        "engine_type": "hybrid"
                    },
                    {
                        "query": "在庫管理システムに関する要望を教えてください",
                        "engine_type": "hybrid"
                    },
                    {
                        "query": "どの会社が配送ルート最適化に興味がありますか？",
                        "engine_type": "summary_only"
                    },
                    {
                        "query": "具体的にどのような発言がありましたか？",
                        "engine_type": "conversation_only"
                    }
                ]
            
            # クエリエンジンの作成
            from .query_engines import ConversationQueryEngineFactory
            from llama_index.core.query_engine import SubQuestionQueryEngine
            from llama_index.core.tools import QueryEngineTool
            
            # 各セッションのインデックスをクエリエンジンに変換
            all_summary_indices = []
            all_conversation_indices = []
            
            self.monitor.log_event("info", f"概要インデックス数: {len(all_summary_indices)}")
            self.monitor.log_event("info", f"会話インデックス数: {len(all_conversation_indices)}")
            
            # 統合クエリエンジンを作成（全セッションを横断検索）
            # 各クエリを実行
            for i, query_info in enumerate(queries, 1):
                self.monitor.log_event("info", f"\n--- クエリ {i}/{len(queries)} ---")
                self.monitor.log_event("info", f"質問: {query_info['query']}")
                self.monitor.log_event("info", f"エンジンタイプ: {query_info['engine_type']}")
                
                engine_type = query_info.get("engine_type", "hybrid")
                results = []
                
                # 各セッションのインデックスに対してクエリを実行
                if engine_type == "hybrid":
                    # 概要と会話の両方を検索
                    for idx in indices:
                        try:
                            query_engine = idx.as_query_engine(similarity_top_k=3)
                            response = query_engine.query(query_info["query"])
                            if response.response and response.response.strip():
                                results.append({
                                    "index": idx_name,
                                    "response": response.response,
                                    "score": response.source_nodes[0].score if response.source_nodes else 0
                                })
                        except Exception as e:
                            self.monitor.log_event("warning", f"  インデックス {idx_name} のクエリエラー: {e}")
                            
                elif engine_type == "summary_only":
                    # 概要のみ検索
                    for idx in indices:
                        try:
                            query_engine = idx.as_query_engine(similarity_top_k=3)
                            response = query_engine.query(query_info["query"])
                            if response.response and response.response.strip():
                                results.append({
                                    "index": idx_name,
                                    "response": response.response,
                                    "score": response.source_nodes[0].score if response.source_nodes else 0
                                })
                        except Exception as e:
                            self.monitor.log_event("warning", f"  インデックス {idx_name} のクエリエラー: {e}")
                            
                elif engine_type == "conversation_only":
                    # 会話のみ検索
                    for idx in indices:
                        try:
                            query_engine = idx.as_query_engine(similarity_top_k=3)
                            response = query_engine.query(query_info["query"])
                            if response.response and response.response.strip():
                                results.append({
                                    "index": idx_name,
                                    "response": response.response,
                                    "score": response.source_nodes[0].score if response.source_nodes else 0
                                })
                        except Exception as e:
                            self.monitor.log_event("warning", f"  インデックス {idx_name} のクエリエラー: {e}")
                
                # スコアでソートして上位結果を表示
                results.sort(key=lambda x: x["score"], reverse=True)
                
                if results:
                    self.monitor.log_event("info", f"回答候補数: {len(results)}")
                    for rank, result in enumerate(results[:5], 1):  # 上位5件表示
                        self.monitor.log_event("info", f"\n  [{rank}] {result['index']} (スコア: {result['score']:.4f})")
                        self.monitor.log_event("info", f"  {result['response']}")
                else:
                    self.monitor.log_event("info", "該当する回答が見つかりませんでした")
                
                self.monitor.log_event("info", "-" * 50)
            
        except Exception as e:
            self.monitor.log_event("error", f"クエリ処理エラー: {e}")
            raise

    def run(self, run_indexing: bool = True, run_query: bool = True):
        """全体テスト実行"""
        try:
            self.monitor.log_event("info", "========================================")
            self.monitor.log_event("info", "会話情報RAGシステム テスト開始")
            self.monitor.log_event("info", "========================================")
            
            # コールバック設定（基底クラスのメソッドを使用）
            self._setup_callback()
            
            # インデクシング実行
            if run_indexing:
                self.run_indexing()
            
            # クエリ実行
            if run_query:
                self.run_query()

            self.monitor.log_event("info", "========================================")
            self.monitor.log_event("info", "全テスト完了")
            self.monitor.log_event("info", "========================================")
            
        except Exception as e:
            self.monitor.log_event("error", f"テスト実行エラー: {e}")
            raise


if __name__ == "__main__":
    test = TranscriptionTest()
    test.run()
