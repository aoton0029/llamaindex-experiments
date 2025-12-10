"""
会話情報RAGシステムのテストスクリプト
"""
import os
import logging
import dotenv
from typing import List, Optional
from ..factories import (
    LLMFactory,
    EmbeddingFactory,
    IndexBuilderFactory,
    RetrieverFactory,
    QueryEngineFactory,
    TemplatePromptSettings,
    DomainLLMSettings,
)
from .loaders import DocumentLoader
from .models import ConversationSession, ConversationChunkMetadata
from .parsers import ConversationParser
from .retrievers import ConversationRetriever
from .query_engines import ConversationQueryEngine
from ..db import DatabaseManager, DatabaseConfig, StorageContextConfig, StorageContextManager, DocstoreConfig, IndexStoreConfig, VectorStoreConfig
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionTestConfig:
    """会話情報RAGシステムの設定"""
    PARSER_TYPE = "hybrid"  # "utterance" | "summary" | "hybrid"
    CHUNK_OVERLAP = 1
    CONTEXT_WINDOW = 3
    SIMILARITY_TOP_K = 10
    EMBED_MODEL = "qwen3-embedding:8b"
    DIM = 4096
    LLM_MODEL = "Qwen3-32B-AWQ"
    LLM_CONFIG = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "additional_kwargs": {
            "frequency_penalty": 2.0,
            "presence_penalty": -1.0
        }
    }
    

class TranscriptionTest:
    """会話情報RAGシステムのテストクラス"""
    
    def __init__(self, config: Optional[TranscriptionTestConfig] = None):
        self.config = config or TranscriptionTestConfig()
        self.loader = None
        self.sessions = []
        self.documents = []
        self.db_manager = None
        self.context_manager = None
        self.storage_context = None  # 共通のStorageContext
        self.indices = {}  # uid -> index のマッピング
        self.query_engine = None
        self._log_event("info", "TranscriptionTest初期化完了")

    def _setup_llm(self):
        """LLMの設定"""
        try:
            llm = LLMFactory.create(
                backend="vllm",
                model=self.config.LLM_MODEL,
                base_url="http://vllm:8000",
                **self.config.LLM_CONFIG
            )
            Settings.llm = llm
            self._log_event("info", f"LLM設定完了: {self.config.LLM_MODEL}")
        except Exception as e:
            self._log_event("error", f"LLM設定エラー: {e}")
            raise

    def _setup_embedding(self):
        """Embeddingモデルの設定"""
        try:
            embed_model = EmbeddingFactory.create(
                backend="ollama",
                model=self.config.EMBED_MODEL,
                base_url="http://ollama:11434",
            )
            Settings.embed_model = embed_model
            self._log_event("info", f"Embedding設定完了: {self.config.EMBED_MODEL}")
        except Exception as e:
            self._log_event("error", f"Embedding設定エラー: {e}")
            raise

    def _setup_database_manager(self):
        """データベースマネージャーの設定"""
        try:
            db_manager = DatabaseManager(DatabaseConfig())
            self._log_event("info", f"DatabaseManager設定完了")
            return db_manager
        except Exception as e:
            self._log_event("error", f"DatabaseManager設定エラー: {e}")
            raise

    def _setup_context_manager(self, db_manager: DatabaseManager):
        """StorageContextManagerの設定"""
        try:
            self.context_manager = StorageContextManager(db_manager=db_manager)
            self._log_event("info", "StorageContextManager設定完了")
        except Exception as e:
            self._log_event("error", f"StorageContextManager設定エラー: {e}")
            raise
    
    def _create_shared_storage_context(self, reset: bool = False):
        """会話情報RAG用の共通StorageContextを作成"""
        try:
            # transcriptionドメイン全体で共通のStorageContext
            docstore = DocstoreConfig(
                namespace="transcription",
                collection_name="transcription_docstore",
            )
            indexstore = IndexStoreConfig(
                namespace="transcription",
                collection_suffix="_transcription",
            )
            vectorstore = VectorStoreConfig(
                collection_name="transcription_conversations",
                dim=self.config.DIM,
                schema=ConversationChunkMetadata.schema(self.config.DIM),
            )
            storage_config = StorageContextConfig(
                context_name="transcription_shared",
                docstore=docstore,
                indexstore=indexstore,
                vectorstore=vectorstore,
            )
            if reset:
                self.context_manager.get_storage_context(storage_config)
                self.context_manager.drop_storage_context(storage_config)
                self._log_event("info", "既存のStorageContextを削除しました")
            
            self.storage_context = self.context_manager.create_storage_context(storage_config)
            self._log_event("info", "共通StorageContext作成完了")
            
        except Exception as e:
            self._log_event("error", f"StorageContext作成エラー: {e}")
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
                self._log_event("info", f"会話セッションをキャッシュから読み込み完了: {len(self.sessions)}件")
            else:
                self.sessions = self.loader.load_data()
                self._log_event("info", f"会話セッション取得完了: {len(self.sessions)}件")
                # キャッシュ保存
                with open(cache_file_path, 'wb') as f:
                    import pickle
                    pickle.dump(self.sessions, f)
                self._log_event("info", f"会話セッションをキャッシュに保存: {cache_file_path}")

            if self.sessions:
                session = self.sessions[0]
                self._log_event("info", f"サンプル: UID={session.uid}, 会社={session.company_name}")
        except Exception as e:
            self._log_event("error", f"データ取得エラー: {e}")
            raise

    def _setup_callback(self):
        from llama_index.core import set_global_handler
        from langfuse.llama_index import LlamaIndexCallbackHandler
        from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
        
        try:
            dotenv.load_dotenv()
            secret_key = os.getenv('LANGFUSE_SECRET_KEY')
            public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
            host = os.getenv('LANGFUSE_BASE_URL')
            logger.info(f"LANGFUSE_SECRET_KEY: {secret_key}")
            logger.info(f"LANGFUSE_PUBLIC_KEY: {public_key}")
            logger.info(f"LANGFUSE_BASE_URL: {host}")
            # set_global_handler("langfuse", public_key=public_key, secret_key=secret_key, host=host)

            langfuse_callback_handler = LlamaIndexCallbackHandler(public_key=public_key, secret_key=secret_key, host=host)
            Settings.callback_manager = CallbackManager([langfuse_callback_handler])
            
            # from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
            # llamadebughandler = LlamaDebugHandler()
            # callback_manager = CallbackManager([llamadebughandler])
            # Settings.callback_manager = callback_manager
        except Exception as e:
            self._log_event("warning", f"Failed to: {e}")
            raise


    def _log_event(self, category: str, message: str):
        logger.info(f"[{category.upper()}] {message}")


    def run_indexing(self):
        """インデクシング処理の実行（会話セッションごと）"""
        try:
            self._log_event("info", "=== インデクシング開始 ===")
            
            # 1. LLM/Embedding設定
            self._setup_llm()
            self._setup_embedding()
            
            # 2. データベース/StorageContext設定
            db_manager = self._setup_database_manager()
            self._setup_context_manager(db_manager)
            self._create_shared_storage_context()
            
            # 3. データ取得
            self._load_data()
            
            if not self.sessions:
                self._log_event("warning", "会話セッションが0件です")
                return None
            
            # 4. Document変換
            self.documents = self.loader.sessions_to_documents(self.sessions)
            self._log_event("info", f"Document変換完了: {len(self.documents)}件")
            
            # 5. チャンキング
            parser = ConversationParser(
                chunk_size=self.config.CHUNK_OVERLAP * 512,
                chunk_overlap=self.config.CHUNK_OVERLAP * 128
            )
            nodes = parser.parse_documents(self.documents)
            self._log_event("info", f"チャンキング完了: {len(nodes)}チャンク")
            
            # 6. VectorStoreIndexの作成（全セッションを統合）
            self._log_event("info", "VectorStoreIndex作成中...")
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=True
            )
            
            # インデックスを保存
            self.indices["shared_index"] = index
            self._log_event("info", f"インデックス作成完了: {len(nodes)}ノード")
            
            # StorageContextを永続化
            self.storage_context.persist()
            self._log_event("info", "StorageContext永続化完了")
                        
            return self.indices
            
        except Exception as e:
            self._log_event("error", f"インデクシングエラー: {e}")
            raise

    def run_query(self, queries: Optional[List[dict]] = None):
        """クエリ処理の実行"""
        try:
            self._log_event("info", "=== クエリ処理開始 ===")
            
            if not self.indices:
                self._log_event("error", "インデックスが作成されていません")
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
                self._log_event("info", f"\n--- クエリ {i}/{len(queries)} ---")
                self._log_event("info", f"質問: {query_info['query']}")
                if query_info.get('filters'):
                    self._log_event("info", f"フィルタ: {query_info['filters']}")
                
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
                self._log_event("info", f"回答:\n{response.response}")
                
                # 参照元情報
                if response.source_nodes:
                    self._log_event("info", f"\n参照元: {len(response.source_nodes)}件")
                    for j, node in enumerate(response.source_nodes[:3], 1):
                        company = node.node.metadata.get('会社名', '不明')
                        uid = node.node.metadata.get('uid', '不明')
                        score = node.score
                        self._log_event("info", f"  {j}. UID={uid}, 会社={company}, スコア={score:.3f}")
                
                self._log_event("info", "-" * 50)
            
        except Exception as e:
            self._log_event("error", f"クエリ処理エラー: {e}")
            raise

    def run(self, run_indexing: bool = True, run_query: bool = True):
        """全体テスト実行"""
        try:
            self._log_event("info", "========================================")
            self._log_event("info", "会話情報RAGシステム テスト開始")
            self._log_event("info", "========================================")
            
            # コールバック設定
            self._setup_callback()
            
            # インデクシング実行
            if run_indexing:
                self.run_indexing()
            
            # クエリ実行
            if run_query:
                self.run_query()

            self._log_event("info", "========================================")
            self._log_event("info", "全テスト完了")
            self._log_event("info", "========================================")
            
        except Exception as e:
            self._log_event("error", f"テスト実行エラー: {e}")
            raise


if __name__ == "__main__":
    test = TranscriptionTest()
    test.run()
