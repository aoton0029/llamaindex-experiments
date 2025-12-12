import os
import sys
import logging
import json
import dotenv
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from llama_index.core import Settings, StorageContext
from llama_index.core.node_parser import NodeParser
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import BaseLLM
from llama_index.core.extractors import BaseExtractor

from transformers import AutoTokenizer
from test_monitor import TestMonitor
from infrastructure import (
    VectorStoreConfig,
    VectorStoreManager,
    DocumentStoreConfig,
    DocumentStoreManager,
    IndexStoreConfig,
    IndexStoreManager,
    GraphStoreConfig,
    GraphStoreManager,
    RelationalStoreConfig,
    RelationalStoreManager,
    StorageContextConfig, 
    StorageContextManager
)


logger = logging.getLogger(__name__)


class TestRunnerBase(ABC):
    """
    テストランナーの基底クラス
    共通のセットアップメソッドを提供
    """

    def __init__(self, result_dir: str = None):
        """
        初期化
        
        Args:
            config_dir: 設定ファイルのディレクトリ
            test_dir: テストパターンディレクトリ
            result_dir: 結果出力ディレクトリ
        """
        self.result_dir = result_dir or os.environ.get('TEST_RESULT_DIR', './test_results')
        self.monitor = TestMonitor(result_dir)
        self.db_manager: RelationalStoreManager = None
        self.storage_context_manager: StorageContextManager = None

    def _setup_llm(
            self, 
            model_name: str, 
            temperature: float, 
            max_tokens: int, 
            presence_penalty: Optional[float] = None, 
            frequency_penalty: Optional[float] = None,
            stop: List[str] = None) -> BaseLLM:
        """LLMをセットアップ"""
        try:
            self.monitor.log_event("setup", "Setting up LLM...")
            
            # 環境変数から設定を取得
            backend = os.getenv('LLM_BACKEND', 'vllm')
            base_url = os.getenv('LLM_BASE_URL')

            additional_params = {}
            if presence_penalty is not None:
                additional_params['presence_penalty'] = presence_penalty
            if frequency_penalty is not None:
                additional_params['frequency_penalty'] = frequency_penalty
            if stop is not None:
                additional_params['stop'] = stop

            llm = LLMFactory.create(
                backend=backend,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                additional_kwargs=additional_params
            )
            self.monitor.log_event("setup", f"Set up LLM: {model_name}")
            return llm
        except Exception as e:
            logger.error(f"LLM setup failed: {e}")
            raise       

    def _setup_embedding(self, model_name: str, dim: int, **kwargs) -> Tuple[BaseEmbedding, int]:
        """
        埋め込みモデルをセットアップ
        
        Args:
            model_name: 埋め込みモデルの名前
            dim: 埋め込みの次元数
        """
        try:
            self.monitor.log_event("setup", "Setting up embedding model...")
            
            # 環境変数から設定を取得
            backend = os.getenv('EMBEDDING_BACKEND', 'ollama')
            base_url = os.getenv('EMBEDDING_BASE_URL')
            
            embedding = EmbeddingFactory.create(
                backend=backend,
                model_name=model_name,
                base_url=base_url,
                dimensions=dim,
                **kwargs
            )
            self.monitor.log_event("setup", f"Set up embedding model: {model_name}")
            return embedding
        except Exception as e:
            logger.error(f"Embedding model setup failed: {e}")
            raise

    def _setup_tokenizer(self, pretrained_model_name_or_path: str, **kwargs):
        """
        トークナイザーをセットアップ
        """
        try:
            self.monitor.log_event("setup", "Setting up tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            self.monitor.log_event("setup", f"Set up tokenizer: {pretrained_model_name_or_path}")
            return tokenizer
        except Exception as e:
            logger.error(f"Tokenizer setup failed: {e}")
            raise

    # def _setup_indexbuilder(
    #     self,
    #     indexing_type: str,
    #     storage_context: StorageContext,
    # ) -> IndexBuilder:
    #     """インデックスビルダーをセットアップ"""
    #     try:
    #         self.monitor.log_event("setup", "Setting up index builder...")
    #         index_builder = IndexBuilderFactory.create(
    #             builder_type=indexing_type,
    #             storage_context=storage_context,
    #             show_progress=True
    #         )
    #         self.monitor.log_event("setup", f"Set up index builder: {indexing_type}")
    #         return index_builder
    #     except Exception as e:
    #         logger.error(f"Index builder setup failed: {e}")
    #         raise

    # def _setup_chunker(self, chunker_type: str, **kwargs) -> NodeParser:
    #     """チャンカーをセットアップ"""
    #     try:
    #         self.monitor.log_event("setup", "Setting up chunker...")
            
    #         nodeparser = ChunkerFactory.create(
    #             chunker_type=chunker_type,
    #             **kwargs
    #         )

    #         self.monitor.log_event("setup", f"Set up chunker: {chunker_type}")
    #         return nodeparser
    #     except Exception as e:
    #         logger.error(f"Chunker setup failed: {e}")
    #         raise


    # def _setup_indexbuilder(
    #     self, 
    #     indexing_type: str, 
    #     storage_context: StorageContext,
    # ) -> IndexBuilder:
    #     """インデックスビルダーをセットアップ"""
    #     try:
    #         self.monitor.log_event("setup", "Setting up index builder...")
    #         index_builder = IndexBuilderFactory.create(
    #             builder_type=indexing_type,
    #             storage_context=storage_context,
    #             show_progress=True
    #         )
    #         self.monitor.log_event("setup", f"Set up index builder: {indexing_type}")
    #         return index_builder
    #     except Exception as e:
    #         logger.error(f"Index builder setup failed: {e}")
    #         raise

    # def _setup_extractor(self, extractor_type: str, **kwargs) -> BaseExtractor:
    #     """エクストラクタをセットアップ
        
    #     Args:
    #         extractor_type: エクストラクタタイプ
    #     """
    #     try:
    #         self.monitor.log_event("setup", "Setting up extractor...")

    #         extractor = ExtractorFactory.create_extractor(
    #             extractor_type=extractor_type, 
    #             **kwargs
    #         )

    #         self.monitor.log_event("setup", f"Set up extractor: {extractor_type}")
    #         return extractor
    #     except Exception as e:
    #         logger.error(f"Extractor setup failed: {e}")
    #         raise

    # def _setup_extractors(self, extractor_configs: List[Dict[str, Any]]) -> List[BaseExtractor]:
    #     """エクストラクタ群をセットアップ"""
    #     try:
    #         self.monitor.log_event("setup", "Setting up extractors...")
    #         extractors = []
    #         for config in extractor_configs:
    #             extractor_type = config.get("extractor_type")
    #             extractor_params = config.get("params", {})
    #             extractor = self._setup_extractor(extractor_type, **extractor_params)
    #             extractors.append(extractor)
    #         self.monitor.log_event("setup", f"Set up {len(extractors)} extractors")
    #         return extractors
    #     except Exception as e:
    #         logger.error(f"Extractors setup failed: {e}")
    #         raise
    
    # def _setup_preprocessor(self) -> BasePreProcessor:
    #     """前処理をセットアップ"""
    #     try:
    #         self.monitor.log_event("setup", "Setting up preprocessor...")
    #         # TODO: 前処理の実装
    #         preprocessor = None  # PreProcessorFactory.create(...)
            
    #         self.monitor.log_event("setup", "Set up preprocessor")
    #         return preprocessor
    #     except Exception as e:
    #         logger.error(f"Preprocessor setup failed: {e}")
    #         raise


    def _setup_prompt_helper(self, tokenizer):
        """プロンプトヘルパーをセットアップ"""
        try:
            from llama_index.core import PromptHelper
            
            self.monitor.log_event("setup", "Setting up prompt helper...")
            
            prompt_helper = PromptHelper(
                context_window=4096,
                num_output=512,
                chunk_overlap_ratio=0.1,
                chunk_size_limit=None,
                separator="。",
                tokenizer=tokenizer.decode
            )
            
            self.monitor.log_event("setup", f"Set up prompt helper:")
            return prompt_helper
        except Exception as e:
            logger.error(f"Prompt helper setup failed: {e}")
            raise

    def _setup_vector_store_manager(self):
        """VectorStoreManagerをセットアップ"""
        try:
            self.monitor.log_event("setup", "Setting up vector store manager...")
            
            # 環境変数から設定を取得
            vector_store_config = VectorStoreConfig(
                host=os.getenv('VECTOR_STORE_HOST', 'localhost'),
                port=int(os.getenv('VECTOR_STORE_PORT', '19530')),
                user=os.getenv('VECTOR_STORE_USERNAME'),
                password=os.getenv('VECTOR_STORE_PASSWORD')
            )
            
            self.vector_store_manager = VectorStoreManager(vector_store_config)
            self.monitor.log_event("setup", "Set up vector store manager")
            return self.vector_store_manager
        except Exception as e:
            logger.error(f"Vector store manager setup failed: {e}")
            raise

    def _setup_document_store_manager(self):
        """DocumentStoreManagerをセットアップ"""
        try:
            self.monitor.log_event("setup", "Setting up document store manager...")
            
            # 環境変数から設定を取得
            document_store_config = DocumentStoreConfig(
                host=os.getenv('DOCUMENT_STORE_HOST', 'localhost'),
                port=int(os.getenv('DOCUMENT_STORE_PORT', '27017')),
                database_name=os.getenv('DOCUMENT_STORE_DATABASE', 'documents'),
                username=os.getenv('DOCUMENT_STORE_USERNAME'),
                password=os.getenv('DOCUMENT_STORE_PASSWORD')
            )
            
            self.document_store_manager = DocumentStoreManager(document_store_config)
            self.monitor.log_event("setup", "Set up document store manager")
            return self.document_store_manager
        except Exception as e:
            logger.error(f"Document store manager setup failed: {e}")
            raise

    def _setup_index_store_manager(self):
        """IndexStoreManagerをセットアップ"""
        try:
            self.monitor.log_event("setup", "Setting up index store manager...")
            
            # 環境変数から設定を取得
            index_store_config = IndexStoreConfig(
                host=os.getenv('INDEX_STORE_HOST', 'localhost'),
                port=int(os.getenv('INDEX_STORE_PORT', '6379')),
                password=os.getenv('INDEX_STORE_PASSWORD'),
                db=int(os.getenv('INDEX_STORE_DB', '0'))
            )
            
            self.index_store_manager = IndexStoreManager(index_store_config)
            self.monitor.log_event("setup", "Set up index store manager")
            return self.index_store_manager
        except Exception as e:
            logger.error(f"Index store manager setup failed: {e}")
            raise

    def _setup_graph_store_manager(self):
        """GraphStoreManagerをセットアップ"""
        try:
            self.monitor.log_event("setup", "Setting up graph store manager...")
            
            # 環境変数から設定を取得
            graph_store_config = GraphStoreConfig(
                uri=os.getenv('GRAPH_STORE_URI', 'bolt://localhost:7687'),
                username=os.getenv('GRAPH_STORE_USERNAME', 'neo4j'),
                password=os.getenv('GRAPH_STORE_PASSWORD'),
                database=os.getenv('GRAPH_STORE_DATABASE', 'neo4j')
            )
            
            self.graph_store_manager = GraphStoreManager(graph_store_config)
            self.monitor.log_event("setup", "Set up graph store manager")
            return self.graph_store_manager
        except Exception as e:
            logger.error(f"Graph store manager setup failed: {e}")
            raise

    def _setup_relational_store_manager(self):
        """RelationalStoreManagerをセットアップ"""
        try:
            self.monitor.log_event("setup", "Setting up relational store manager...")
            
            # 環境変数から設定を取得
            relational_store_config = RelationalStoreConfig(
                backend=os.getenv('RELATIONAL_STORE_BACKEND', 'sqlserver'),
                server=os.getenv('RELATIONAL_STORE_SERVER', 'sql001'),
                database=os.getenv('RELATIONAL_STORE_DATABASE', 'seizo'),
                username=os.getenv('RELATIONAL_STORE_USERNAME', 'sa'),
                password=os.getenv('RELATIONAL_STORE_PASSWORD'),
                port=int(os.getenv('RELATIONAL_STORE_PORT', '1433')),
            )
            
            self.db_manager = RelationalStoreManager(relational_store_config)
            self.monitor.log_event("setup", "Set up relational store manager")
            return self.db_manager
        except Exception as e:
            logger.error(f"Relational store manager setup failed: {e}")
            raise
    
    
    def _setup_storage_context_manager(
        self,
        vector_store_manager: VectorStoreManager = None,
        document_store_manager: DocumentStoreManager = None,
        index_store_manager: IndexStoreManager = None,
        graph_store_manager: GraphStoreManager = None,
    ) -> StorageContextManager:
        """StorageContextManagerをセットアップ"""
        try:
            self.monitor.log_event("setup", "Setting up storage context manager...")

            if self.storage_context_manager is not None:
                self.monitor.log_event("setup", "Storage context manager already initialized")
                return self.storage_context_manager

            self.storage_context_manager = StorageContextManager(
                vector_store_manager=vector_store_manager,
                document_store_manager=document_store_manager,
                index_store_manager=index_store_manager,
                graph_store_manager=graph_store_manager,
            )

            self.monitor.log_event("setup", "Set up storage context manager")
            return self.storage_context_manager
        except Exception as e:
            logger.error(f"Storage context manager setup failed: {e}")
            raise
    
    def _create_storage_context(
        self, 
        storage_context_config: StorageContextConfig,
        storage_context_manager: StorageContextManager = None,
        drop_existing: bool = False
    ) -> StorageContext:
        """
        StorageContextをセットアップ
        
        Args:
            storage_config_dict: StorageContext設定辞書
            storage_context_manager: StorageContextManagerインスタンス（Noneの場合は既存のものを使用）
            drop_existing: 既存のStorageContextを削除してから作成するか
        Returns:
            StorageContextインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up storage context...")
            
            if storage_context_manager is None:
                if self.storage_context_manager is None:
                    self.monitor.log_event("setup", "Storage context manager not found, creating new one...")
                    storage_context_manager = self._setup_storage_context_manager()
                else:
                    storage_context_manager = self.storage_context_manager
            
            # 既存のStorageContextを削除
            if drop_existing:
                self.monitor.log_event("setup", f"Dropping existing storage context... {storage_context_config.context_name}")
                try:
                    storage_context_manager.drop_storage_context(storage_context_config)
                    self.monitor.log_event("setup", f"Dropped existing storage context: {storage_context_config.context_name}")
                except Exception as e:
                    self.monitor.log_event("setup", f"No existing storage context to drop or error: {e}")
            
            # StorageContextを作成
            storage_context = storage_context_manager.create_storage_context(storage_context_config)

            self.monitor.log_event("setup", f"Set up storage context: {storage_context_config.context_name}")
            return storage_context
        except Exception as e:
            logger.error(f"Storage context setup failed: {e}")
            raise
    
    def _drop_storage_context(self, context_name: str):
        """StorageContextを削除"""
        try:
            self.monitor.log_event("cleanup", f"Dropping storage context: {context_name}")
            
            if self.storage_context_manager is None:
                logger.warning("Storage context manager not initialized")
                return
            
            self.storage_context_manager.drop_storage_context_by_name(context_name)
            self.monitor.log_event("cleanup", f"Dropped storage context: {context_name}")
        except Exception as e:
            logger.error(f"Failed to drop storage context: {e}")
            raise

    def _get_file_paths_from_data_source(self, directory: str, file_pattern: str) -> List[Path]:
        """データソース設定からファイルパスのリストを取得"""
        try:
            import glob
            dir_path = Path(directory)
            self.monitor.log_event("data_source", f"Accessing data source directory: {dir_path}")

            # ディレクトリパスを絶対パスに変換
            if not dir_path.is_absolute():
                # 相対パスの場合、プロジェクトルートからの相対パスとして解決
                dir_path = dir_path.resolve()
            else:
                dir_path = dir_path

            if not dir_path.exists():
                logger.warning(f"Directory does not exist: {directory}")
                return []
            
            # ファイルパターンに一致するファイルを検索
            pattern = str(directory / file_pattern)
            file_paths = [Path(p) for p in glob.glob(pattern, recursive=False)]

            self.monitor.log_event("data_source", f"Found {len(file_paths)} files in {directory}")
            return sorted(file_paths)
        except Exception as e:
            logger.error(f"Failed to get file paths from data source: {e}")
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
            self.monitor.log_event("warning", f"Failed to: {e}")
            raise



class LLMFactory:
    @staticmethod
    def create(backend: str, model_name: str, base_url: str, **kwargs) -> BaseLLM:
        if backend == "ollama":
            from llama_index.llms.ollama import Ollama
            return Ollama(
                model=model_name, 
                base_url=base_url,
                **kwargs)
        elif backend == "vllm":
            from llama_index.llms.openai_like import OpenAILike
            return OpenAILike(
                model=model_name, 
                api_base=base_url,
                **kwargs)
        raise ValueError(f"Unsupported LLM backend: {backend}")


class EmbeddingFactory:
    @staticmethod
    def create(backend: str, model_name: str, base_url: str, **kwargs) -> BaseEmbedding:
        if backend == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            return OllamaEmbedding(
                model_name=model_name,
                base_url=base_url,
                **kwargs)
        elif backend == "vllm":
            from llama_index.embeddings.openai_like import OpenAILikeEmbedding
            return OpenAILikeEmbedding(
                model_name=model_name, 
                api_base=base_url, 
                **kwargs)

        raise ValueError(f"Unsupported Embedding backend: {backend}")