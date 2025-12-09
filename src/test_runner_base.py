import os
import sys
import logging
import json
import dotenv
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from pathlib import Path
from llama_index.core import Settings, StorageContext
from llama_index.core.node_parser import NodeParser
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import BaseLLM
from llama_index.core.extractors import BaseExtractor
from transformers import AutoTokenizer
from config_manager import ConfigManager
from test_pattern_manager import TestPatternManager
from test_monitor import TestMonitor
from factories import (
    LLMFactory,
    EmbeddingFactory,
    ChunkerFactory,
    IndexBuilderFactory,
    IndexBuilder,
    ExtractorFactory,
    PreProcessorFactory,
    BasePreProcessor,
    IndexMetadataExtractor,
    TemplatePromptSettings,
    DomainLLMSettings,
)
from db import (
    DatabaseManager,
    DatabaseConfig,
    StorageContextConfig,
    StorageContextManager,
)


logger = logging.getLogger(__name__)


class TestRunnerBase(ABC):
    """
    テストランナーの基底クラス
    共通のセットアップメソッドを提供
    """

    def __init__(self, config_dir: str, test_dir: str, result_dir: str):
        """
        初期化
        
        Args:
            config_dir: 設定ファイルのディレクトリ
            test_dir: テストパターンディレクトリ
            result_dir: 結果出力ディレクトリ
        """
        self.config_dir = config_dir
        self.result_dir = result_dir
        self.test_dir = test_dir
        self.config_manager = ConfigManager(config_dir)
        self.test_pattern_manager = TestPatternManager(test_dir, self.config_manager)
        self.monitor = TestMonitor(result_dir)
        self.db_manager: DatabaseManager = None
        self.storage_context_manager: StorageContextManager = None
        self.index_metadata_extractor: IndexMetadataExtractor = None

    def _setup_template_prompts(self):
        """
        テンプレートプロンプト設定をセットアップ
        """
        try:
            self.monitor.log_event("setup", "Setting up template prompt settings...")
            TemplatePromptSettings.initialize(self.config_manager.get_template_prompts())
            self.monitor.log_event("setup", "Set up template prompt settings")
        except Exception as e:
            logger.error(f"Template prompt settings setup failed: {e}")
            raise

    def _setup_llm(self, llm_config_model_name: str) -> BaseLLM:
        """
        LLMをセットアップ
        
        Args:
            llm_config_model_name: LLM設定モデル名
        """
        try:
            self.monitor.log_event("setup", "Setting up domain-specific LLMs...")
            llm_config = self.config_manager.get_llm_config(llm_config_model_name)
            domain_configs = self.config_manager.get_llm_domain_config()
            DomainLLMSettings.initialize(llm_config, domain_configs)
            self.monitor.log_event("setup", "Set up domain-specific LLMs")
            return DomainLLMSettings.default()
        except Exception as e:
            logger.error(f"LLM setup failed: {e}")
            raise        

    def _setup_embedding(self, embedding_config_model_name: str) -> Tuple[BaseEmbedding, int]:
        """
        埋め込みモデルをセットアップ
        
        Args:
            embedding_config_model_name: 埋め込みモデル設定名
            
        Returns:
            (埋め込みモデルインスタンス, 次元数)のタプル
        """
        try:
            self.monitor.log_event("setup", "Setting up embedding model...")
            embedding_config = self.config_manager.get_embedding_config(embedding_config_model_name)
            
            backend = embedding_config["backend"]
            model_name = embedding_config["model_name"]
            base_url = embedding_config.get("base_url")
            dim = embedding_config["dimensions"]

            embedding = EmbeddingFactory.create(
                backend=backend,
                model_name=model_name,
                base_url=base_url,
                **embedding_config.get("kwargs", {})
            )
            self.monitor.log_event("setup", f"Set up embedding model: {model_name}")
            return embedding, dim
        except Exception as e:
            logger.error(f"Embedding model setup failed: {e}")
            raise

    def _setup_tokenizer(self, tokenizer_config_model_name: str):
        """
        トークナイザーをセットアップ
        
        Args:
            tokenizer_config_model_name: トークナイザー設定モデル名
            
        Returns:
            トークナイザーインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up tokenizer...")
            tokenizer_config = self.config_manager.get_tokenizer_config(tokenizer_config_model_name)
            
            model_name = tokenizer_config["model_name"]
            tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config['kwargs'])
            self.monitor.log_event("setup", f"Set up tokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Tokenizer setup failed: {e}")
            raise

    def _setup_indexbuilder(
        self, 
        indexing_type: str, 
        storage_context: StorageContext,
    ) -> IndexBuilder:
        """
        インデックスビルダーをセットアップ
        
        Args:
            indexing_type: インデックスタイプ
            storage_context: ストレージコンテキスト
            llm: LLMインスタンス（オプション）
            
        Returns:
            IndexBuilderインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up index builder...")
            index_builder = IndexBuilderFactory.create(
                builder_type=indexing_type,
                storage_context=storage_context,
                show_progress=True
            )
            self.monitor.log_event("setup", f"Set up index builder: {indexing_type}")
            return index_builder
        except Exception as e:
            logger.error(f"Index builder setup failed: {e}")
            raise

    def _setup_chunker(self, chunking_config_model_name: str) -> NodeParser:
        """
        チャンカーをセットアップ
        
        Args:
            chunking_config_model_name: チャンキング設定モデル名
            
        Returns:
            NodeParserインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up chunker...")
            chunking_config = self.config_manager.get_chunking_config(chunking_config_model_name)
            
            chunker_type = chunking_config['type']
            nodeparser = ChunkerFactory.create(
                chunker_type=chunker_type,
                **chunking_config.get("kwargs", {})
            )
            self.monitor.log_event("setup", f"Set up chunker: {chunker_type}")
            return nodeparser
        except Exception as e:
            logger.error(f"Chunker setup failed: {e}")
            raise
    
    def _setup_prompt_helper(self, tokenizer, prompt_helper_config_model_name: str):
        """
        プロンプトヘルパーをセットアップ
        
        Args:
            prompt_helper_config_model_name: プロンプトヘルパー設定モデル名
            
        Returns:
            PromptHelperインスタンス
        """
        try:
            from llama_index.core import PromptHelper
            
            self.monitor.log_event("setup", "Setting up prompt helper...")
            prompt_helper_config = self.config_manager.get_prompt_helper_config(prompt_helper_config_model_name)
            
            prompt_helper = PromptHelper(
                context_window=prompt_helper_config.get("context_window", 4096),
                num_output=prompt_helper_config.get("num_output", 512),
                chunk_overlap_ratio=prompt_helper_config.get("chunk_overlap_ratio", 0.1),
                chunk_size_limit=prompt_helper_config.get("chunk_size_limit"),
                separator=prompt_helper_config.get("separator", " "),
                tokenizer=tokenizer.decode
            )
            
            self.monitor.log_event("setup", f"Set up prompt helper: {prompt_helper_config_model_name}")
            return prompt_helper
        except Exception as e:
            logger.error(f"Prompt helper setup failed: {e}")
            raise
    
    def _setup_extractors(self, extractor_pattern_name: str) -> List[BaseExtractor]:
        """
        複数のエクストラクタをセットアップ
        
        Args:
            extractor_pattern_name: エクストラクタパターン名
            
        Returns:
            エクストラクタインスタンスのリスト
        """
        try:
            self.monitor.log_event("setup", "Setting up extractors...")
            extractor_pattern_config = self.config_manager.get_extractor_pattern_config(extractor_pattern_name)
            extractor_configs = extractor_pattern_config.get("extractors", [])
            
            extractors = []
            for config in extractor_configs:
                extractor_type = config['type']
                extractor = ExtractorFactory.create_extractor(
                    extractor_type=extractor_type,
                    **config.get("kwargs", {})
                )
                extractors.append(extractor)
                self.monitor.log_event("setup", f"Set up extractor: {extractor_type}")
            return extractors
        except Exception as e:
            logger.error(f"Extractor setup failed: {e}")
            raise
    
    def _setup_preprocessor(self) -> BasePreProcessor:
        """
        前処理をセットアップ
        
        Returns:
            前処理パイプライン
        """
        try:
            self.monitor.log_event("setup", "Setting up preprocessor...")
            # TODO: 前処理の実装
            preprocessor = None  # PreProcessorFactory.create(...)
            
            self.monitor.log_event("setup", "Set up preprocessor")
            return preprocessor
        except Exception as e:
            logger.error(f"Preprocessor setup failed: {e}")
            raise
    
    def _setup_database_manager(self, db_config_dict: Dict[str, Any] = None) -> DatabaseManager:
        """
        DatabaseManagerをセットアップ
        
        Args:
            db_config_dict: データベース設定辞書（Noneの場合はデフォルト設定を使用）
            
        Returns:
            DatabaseManagerインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up database manager...")
            
            if db_config_dict:
                db_config = DatabaseConfig(**db_config_dict)
                self.db_manager = DatabaseManager(config=db_config)
            else:
                self.db_manager = DatabaseManager()
            
            # 接続テスト
            self.db_manager.connect_all()
            health_status = self.db_manager.health_check_all()
            
            for db_name, status in health_status.items():
                if status:
                    self.monitor.log_event("setup", f"Database {db_name}: OK")
                else:
                    self.monitor.log_event("setup", f"Database {db_name}: FAILED", level="warning")
            
            self.monitor.log_event("setup", "Set up database manager")

            self.index_metadata_extractor = IndexMetadataExtractor(self.db_manager.get_milvus_client())
            self.monitor.log_event("setup", "Set up index metadata extractor")

            return self.db_manager
        except Exception as e:
            logger.error(f"Database manager setup failed: {e}")
            raise
    
    def _setup_storage_context_manager(self, db_manager: DatabaseManager = None) -> StorageContextManager:
        """
        StorageContextManagerをセットアップ
        
        Args:
            db_manager: DatabaseManagerインスタンス（Noneの場合は既存のものを使用）
            
        Returns:
            StorageContextManagerインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up storage context manager...")
            
            if db_manager is None:
                if self.db_manager is None:
                    self.monitor.log_event("setup", "Database manager not found, creating new one...")
                    db_manager = self._setup_database_manager()
                else:
                    db_manager = self.db_manager
            
            self.storage_context_manager = StorageContextManager(db_manager)
            
            self.monitor.log_event("setup", "Set up storage context manager")
            return self.storage_context_manager
        except Exception as e:
            logger.error(f"Storage context manager setup failed: {e}")
            raise
    
    def _setup_storage_context(
        self, 
        storage_config_dict: Dict[str, Any],
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
            
            # 設定辞書からStorageContextConfigを作成
            storage_config = StorageContextConfig.from_dict(storage_config_dict)
            context_name = storage_config.context_name
            
            # 既存のStorageContextを削除
            if drop_existing:
                self.monitor.log_event("setup", f"Dropping existing storage context... {context_name}")
                try:
                    storage_context_manager.drop_storage_context(storage_config)
                    self.monitor.log_event("setup", f"Dropped existing storage context: {context_name}")
                except Exception as e:
                    self.monitor.log_event("setup", f"No existing storage context to drop or error: {e}")
            
            # StorageContextを作成
            storage_context = storage_context_manager.create_storage_context(storage_config)
            
            self.monitor.log_event("setup", f"Set up storage context: {context_name}")
            return storage_context
        except Exception as e:
            logger.error(f"Storage context setup failed: {e}")
            raise
    
    def _drop_storage_context(self, context_name: str):
        """
        StorageContextを削除
        
        Args:
            context_name: StorageContext名
        """
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
    
    def _get_file_paths_from_data_source(self, data_source: Dict[str, Any]) -> List[Path]:
        """
        データソース設定からファイルパスのリストを取得
        
        Args:
            data_source: データソース設定辞書
                - type: データソースタイプ（"pdf", "markdown"など）
                - directory: ディレクトリパス
                - file_pattern: ファイルパターン（例: "*.pdf", "*.md"）
                
        Returns:
            ファイルパスのリスト
        """
        try:
            import glob
            
            directory = data_source.get("directory", "")
            file_pattern = data_source.get("file_pattern", "*.*")
            
            # ディレクトリパスを絶対パスに変換
            if not Path(directory).is_absolute():
                # 相対パスの場合、プロジェクトルートからの相対パスとして解決
                directory = Path(directory).resolve()
            else:
                directory = Path(directory)
            
            if not directory.exists():
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



