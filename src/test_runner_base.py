import sys
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from pathlib import Path
from llama_index.core import Settings, StorageContext
from llama_index.core.node_parser import NodeParser
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import BaseLLM
from llama_index.core.extractors import BaseExtractor
from transformers import AutoTokenizer
from services.config_manager import ConfigManager
from test_monitor import TestMonitor
from factories import (
    LLMFactory,
    EmbeddingFactory,
    ChunkerFactory,
    IndexBuilderFactory,
    IndexBuilder,
    ExtractorFactory,
    PreProcessorFactory,
    BasePreProcessor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
llamadebughandler = LlamaDebugHandler()
callback_manager = CallbackManager([llamadebughandler])
Settings.callback_manager = callback_manager


class TestRunnerBase(ABC):
    """
    テストランナーの基底クラス
    共通のセットアップメソッドを提供
    """
    
    def __init__(self, config_dir: str, data_dir: str, result_dir: str):
        """
        初期化
        
        Args:
            config_dir: 設定ファイルのディレクトリ
            data_dir: データファイルのディレクトリ
            result_dir: 結果出力ディレクトリ
        """
        self.config_dir = config_dir
        # self.data_dir = data_dir
        self.result_dir = result_dir
        self.config_manager = ConfigManager(config_dir)
        self.monitor = TestMonitor(result_dir)
    
    def _setup_llm(self, llm_config: Dict[str, Any]) -> BaseLLM:
        """
        LLMをセットアップ
        
        Args:
            llm_config: LLM設定
            
        Returns:
            LLMインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up LLM...")
            backend = llm_config["backend"]
            base_url = llm_config["base_url"]
            model_name = llm_config["model_name"]
            
            llm = LLMFactory.create(
                backend=backend,
                model_name=model_name,
                base_url=base_url,
                **llm_config.get("kwargs", {})
            )

            self.monitor.log_event("setup", f"Set up LLM: {model_name}")
            return llm
        except Exception as e:
            logger.error(f"LLM setup failed: {e}")
            raise

    def _setup_embedding(self, embedding_config: Dict[str, Any]) -> Tuple[BaseEmbedding, int]:
        """
        埋め込みモデルをセットアップ
        
        Args:
            embedding_config: 埋め込みモデル設定
            
        Returns:
            (埋め込みモデルインスタンス, 次元数)のタプル
        """
        try:
            self.monitor.log_event("setup", "Setting up embedding model...")
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

    def _setup_tokenizer(self, tokenizer_config: Dict[str, Any]):
        """
        トークナイザーをセットアップ
        
        Args:
            tokenizer_config: トークナイザー設定
            
        Returns:
            トークナイザーインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up tokenizer...")
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
        storage_context: StorageContext
    ) -> IndexBuilder:
        """
        インデックスビルダーをセットアップ
        
        Args:
            indexing_type: インデックスタイプ
            storage_context: ストレージコンテキスト
            
        Returns:
            IndexBuilderインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up index builder...")
            # indexing_config = self.config_manager.get_config("indexing")
            # index_models = indexing_config.get("indexing_config_models", {})
            # pattern_config = index_models.get(indexing_type, {})
            
            # builder_type = pattern_config.get("type")
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

    def _setup_chunker(self, chunking_config: Dict[str, Any]) -> NodeParser:
        """
        チャンカーをセットアップ
        
        Args:
            chunking_config: チャンキング設定
            
        Returns:
            NodeParserインスタンス
        """
        try:
            self.monitor.log_event("setup", "Setting up chunker...")
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
    
    def _setup_extractors(self, extractor_configs: list) -> List[BaseExtractor]:
        """
        複数のエクストラクタをセットアップ
        
        Args:
            extractor_configs: エクストラクタ設定のリスト
            
        Returns:
            エクストラクタインスタンスのリスト
        """
        try:
            self.monitor.log_event("setup", "Setting up extractors...")
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
    
    def _setup_preprocessor(self, use_schema_based: bool = True) -> BasePreProcessor:
        """
        前処理をセットアップ
        
        Args:
            use_schema_based: スキーマベースの前処理を使用するか
            
        Returns:
            前処理パイプライン
        """
        try:
            self.monitor.log_event("setup", "Setting up preprocessor...")
            
            if use_schema_based:
                # スキーマ設定を読み込み
                schema_config = self.config_manager.get_config("schema")
                vector_store_schema = schema_config.get("vector_store_schema", [])
                
                # スキーマベースのパイプラインを作成
                preprocessor = PreProcessorFactory.create_schema_based_pipeline(
                    schema_config=vector_store_schema
                )
            else:
                # デフォルトのパイプラインを作成
                preprocessor = PreProcessorFactory.create_default_pipeline()
            
            self.monitor.log_event("setup", "Set up preprocessor")
            return preprocessor
        except Exception as e:
            logger.error(f"Preprocessor setup failed: {e}")
            raise
    
    def _save_phase_result(self, experiment_dir: Path, phase_name: str, data: Dict[str, Any]):
        """
        各フェーズの結果を保存
        
        Args:
            experiment_dir: 実験ディレクトリ
            phase_name: フェーズ名
            data: 保存するデータ
        """
        phase_dir = experiment_dir / "results" / "phases"
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で保存
        json_path = phase_dir / f"{phase_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved phase result: {phase_name}")
    
    @abstractmethod
    def run_test(self, pattern_name: str):
        """
        実験を実行（サブクラスで実装）
        
        Args:
            pattern_name: 実験パターン名
        """
        pass

    @abstractmethod
    def run(self):
        pass
