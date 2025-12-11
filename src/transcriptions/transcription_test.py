"""
会話情報RAGシステムのテストスクリプト
"""
import os
import logging
import dotenv
from typing import List, Optional, Dict, Any
from pathlib import Path
from src.runners.test_runner_base import TestRunnerBase
from src.adapters.llamaindex.settings import SettingsManager
from .loaders import DocumentLoader
from .models import ConversationSession, ConversationChunkMetadata
from .parsers import ConversationParser
from .retrievers import ConversationRetriever
from .query_engines import ConversationQueryEngine
from ..db import StorageContextConfig, DocstoreConfig, IndexStoreConfig, VectorStoreConfig
from llama_index.core import Settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionTestConfig:
    """会話情報RAGシステムの設定"""
    # ディレクトリ設定
    CONFIG_DIR = str(Path(__file__).parent.parent / "adapters" / "llamaindex" / "settings")
    TEST_DIR = str(Path(__file__).parent / "tests")
    RESULT_DIR = str(Path(__file__).parent / "results")
    
    # LLM設定
    LLM_MODEL_NAME = "vllm_qwen3_32b_awq"  # llms.yamlのキー
    LLM_DOMAIN_NAME = "default"  # ドメイン別LLM設定
    
    # Embedding設定
    EMBEDDING_BACKEND = "ollama"
    EMBEDDING_MODEL = "qwen3-embedding:8b"
    EMBEDDING_BASE_URL = "http://ollama:11434"
    EMBEDDING_DIM = 4096
    
    # Parser設定
    PARSER_TYPE = "hybrid"  # "utterance" | "summary" | "hybrid"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    
    # Retriever設定
    SIMILARITY_TOP_K = 10
    
    # Storage設定
    STORAGE_CONTEXT_NAME = "transcription_shared"
    DOCSTORE_NAMESPACE = "transcription"
    DOCSTORE_COLLECTION = "transcription_docstore"
    INDEXSTORE_NAMESPACE = "transcription"
    INDEXSTORE_SUFFIX = "_transcription"
    VECTORSTORE_COLLECTION = "transcription_conversations"


class TranscriptionTest(TestRunnerBase):
    """会話情報RAGシステムのテストクラス"""
    
    def __init__(self, config: Optional[TranscriptionTestConfig] = None):
        self.config = config or TranscriptionTestConfig()
        
        # 基底クラスの初期化
        super().__init__(
            config_dir=self.config.CONFIG_DIR,
            test_dir=self.config.TEST_DIR,
            result_dir=self.config.RESULT_DIR
        )
        
        # 追加の属性
        self.loader = None
        self.sessions = []
        self.documents = []
        self.storage_context = None  # 共通のStorageContext
        self.indices = {}  # uid -> index のマッピング
        self.query_engine = None
        
        self.monitor.log_event("info", "TranscriptionTest初期化完了")


    
    def _create_shared_storage_context(self, reset: bool = False):
        """会話情報RAG用の共通StorageContextを作成"""
        try:
            # transcriptionドメイン全体で共通のStorageContext設定
            storage_config_dict = {
                "context_name": self.config.STORAGE_CONTEXT_NAME,
                "docstore": {
                    "namespace": self.config.DOCSTORE_NAMESPACE,
                    "collection_name": self.config.DOCSTORE_COLLECTION,
                },
                "indexstore": {
                    "namespace": self.config.INDEXSTORE_NAMESPACE,
                    "collection_suffix": self.config.INDEXSTORE_SUFFIX,
                },
                "vectorstore": {
                    "collection_name": self.config.VECTORSTORE_COLLECTION,
                    "dim": self.config.EMBEDDING_DIM,
                    "schema": ConversationChunkMetadata.schema(self.config.EMBEDDING_DIM),
                }
            }
            
            # 基底クラスのメソッドを使用してStorageContextを作成
            self.storage_context = self._setup_storage_context(
                storage_config_dict=storage_config_dict,
                storage_context_manager=self.storage_context_manager,
                drop_existing=reset
            )
            
            self.monitor.log_event("info", "共通StorageContext作成完了")
            
        except Exception as e:
            self.monitor.log_event("error", f"StorageContext作成エラー: {e}")
            raise

    def _load_data(self):
        """会話セッションデータの取得"""
        try:
            self.loader = DocumentLoader()
            cache_file_path = "conversation_sessions_cache.pkl"
            if os.path.exists(cache_file_path):
                import pickle
                with open(cache_file_path, 'rb') as f:
                    self.sessions = pickle.load(f)
                self.monitor.log_event("info", f"会話セッションをキャッシュから読み込み完了: {len(self.sessions)}件")
            else:
                self.sessions = self.loader.load_data()
                self.monitor.log_event("info", f"会話セッション取得完了: {len(self.sessions)}件")
                # キャッシュ保存
                with open(cache_file_path, 'wb') as f:
                    import pickle
                    pickle.dump(self.sessions, f)
                self.monitor.log_event("info", f"会話セッションをキャッシュに保存: {cache_file_path}")

            if self.sessions:
                session = self.sessions[0]
                self.monitor.log_event("info", f"サンプル: UID={session.uid}, 会社={session.company_name}")
        except Exception as e:
            self.monitor.log_event("error", f"データ取得エラー: {e}")
            raise


    def run_indexing(self):
        """インデクシング処理の実行（会話セッションごと）"""
        try:
            self.monitor.log_event("info", "=== インデクシング開始 ===")
            
            # 1. Settings設定（LLM/Embedding）
            self._setup_settings(self.config.LLM_MODEL_NAME)
            
            # LLMの設定
            llm = self._setup_domain_llm(self.config.LLM_DOMAIN_NAME)
            Settings.llm = llm
            
            # Embeddingの設定
            embed_model, dim = self._setup_embedding(
                backend=self.config.EMBEDDING_BACKEND,
                model_name=self.config.EMBEDDING_MODEL,
                base_url=self.config.EMBEDDING_BASE_URL,
                dim=self.config.EMBEDDING_DIM
            )
            Settings.embed_model = embed_model
            
            # 2. データベース/StorageContext設定
            self._setup_database_manager()
            self._setup_storage_context_manager(self.db_manager)
            self._create_shared_storage_context()
            
            # 3. データ取得
            self._load_data()
            
            if not self.sessions:
                self.monitor.log_event("warning", "会話セッションが0件です")
                return None
            
            # 4. Document変換
            self.documents = self.loader.sessions_to_documents(self.sessions)
            self.monitor.log_event("info", f"Document変換完了: {len(self.documents)}件")
            
            # 5. チャンキング
            parser = ConversationParser(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            nodes = parser.parse_documents(self.documents)
            self.monitor.log_event("info", f"チャンキング完了: {len(nodes)}チャンク")
            
            # 6. VectorStoreIndexの作成（全セッションを統合）
            self.monitor.log_event("info", "VectorStoreIndex作成中...")
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=True
            )
            
            # インデックスを保存
            self.indices["shared_index"] = index
            self.monitor.log_event("info", f"インデックス作成完了: {len(nodes)}ノード")
            
            # StorageContextを永続化
            self.storage_context.persist()
            self.monitor.log_event("info", "StorageContext永続化完了")
                        
            return self.indices
            
        except Exception as e:
            self.monitor.log_event("error", f"インデクシングエラー: {e}")
            raise

    def run_query(self, queries: Optional[List[dict]] = None):
        """クエリ処理の実行"""
        try:
            self.monitor.log_event("info", "=== クエリ処理開始 ===")
            
            if not self.indices:
                self.monitor.log_event("error", "インデックスが作成されていません")
                return
            
            # デフォルトクエリ
            if queries is None:
                queries = [
                    {
                        "query": "ABC商事はどのようなシステムを求めていますか？",
                        "filters": None
                    },
                    {
                        "query": "在庫管理システムに関する要望を教えてください",
                        "filters": {"会社名": "ABC商事株式会社"}
                    },
                    {
                        "query": "決定事項を教えてください",
                        "filters": None
                    },
                    {
                        "query": "配送ルートの最適化について相談している会社はどこですか？",
                        "filters": None
                    }
                ]
            
            index = self.indices["shared_index"]
            
            # 各クエリを実行
            for i, query_info in enumerate(queries, 1):
                self.monitor.log_event("info", f"\n--- クエリ {i}/{len(queries)} ---")
                self.monitor.log_event("info", f"質問: {query_info['query']}")
                if query_info.get('filters'):
                    self.monitor.log_event("info", f"フィルタ: {query_info['filters']}")
                
                # Retrieverの作成
                retriever = ConversationRetriever(
                    index=index,
                    similarity_top_k=self.config.SIMILARITY_TOP_K,
                    filters=query_info.get('filters')
                )
                
                # QueryEngineの作成
                query_engine = ConversationQueryEngine.create(
                    retriever=retriever,
                    response_mode="compact",
                    use_custom_prompt=True
                )
                
                # クエリ実行
                response = query_engine.query(query_info['query'])
                
                # 結果表示
                self.monitor.log_event("info", f"回答:\n{response.response}")
                
                # 参照元情報
                if response.source_nodes:
                    self.monitor.log_event("info", f"\n参照元: {len(response.source_nodes)}件")
                    for j, node in enumerate(response.source_nodes[:3], 1):
                        company = node.node.metadata.get('会社名', '不明')
                        uid = node.node.metadata.get('uid', '不明')
                        score = node.score
                        self.monitor.log_event("info", f"  {j}. UID={uid}, 会社={company}, スコア={score:.3f}")
                
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
