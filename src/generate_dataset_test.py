import logging
import sys
import os
print(os.environ["TIKTOKEN_CACHE_DIR"])
import json
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from llama_index.core import Settings, StorageContext
from llama_index.core.indices.base import BaseIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from db.database_manager import DatabaseConfig, DatabaseManager
from transformers import AutoTokenizer
from services.config_manager import ConfigManager
from test_runner_base import TestRunnerBase
from factories.template_prompts import TemplatePromptSettings
from factories import (
    LlamaIndexDatasetFactory,
    RagasDatasetFactory,
    LLMFactory,
    EmbeddingFactory,
    DocumentLoader,
)

logger = logging.getLogger(__name__)


class DatasetGenerateTestResult(BaseModel):
    """データセット生成テスト結果"""
    success: bool
    message: str
    data: Dict[str, Any]
    experiment_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None


class DatasetGenerateTestRunner(TestRunnerBase):
    """データセット生成テストを実行するクラス"""
    
    def run_test(self, pattern_name: str) -> DatasetGenerateTestResult:
        # テストパターンを取得（現状は評価テストパターンを使用）
        pattern = self.config_manager.get_test_pattern(pattern_name, "evaluation")
        
        # 実験開始
        experiment_id = self.monitor.start_test(pattern_name, pattern)
        experiment_dir = self.monitor.get_test_dir()
        
        try:
            # テンプレートプロンプトを初期化して記録
            self.monitor.log_event("setup", "Recording template prompts")
            TemplatePromptSettings.initialize(self.config_manager)
            TemplatePromptSettings._load_templates()
            template_info = TemplatePromptSettings.get_templates_info()
            self._save_phase_result(experiment_dir, "phase0_template_prompts", template_info)
            
            # トークナイザーの設定
            self.monitor.log_event("setup", "Setting up tokenizer")
            tokenizer_config = self.config_manager.get_tokenizer_config_from_pattern(pattern_name, "evaluation")
            Settings.tokenizer = self._setup_tokenizer(tokenizer_config)
            
            # LLMの設定
            self.monitor.log_event("setup", "Setting up LLM")
            llm_config = self.config_manager.get_llm_config_from_pattern(pattern_name, "evaluation")
            Settings.llm = self._setup_llm(llm_config)
            
            # 埋め込みモデルの設定
            self.monitor.log_event("setup", "Setting up embedding model")
            embedding_config = self.config_manager.get_embedding_config_from_pattern(pattern_name, "evaluation")
            Settings.embed_model, dim = self._setup_embedding(embedding_config)
            
            # ドキュメントのロード
            self.monitor.log_event("processing", "Loading documents")
            document_loader = DocumentLoader()
            all_documents = document_loader.load_from_directory(self.data_dir)
            
            # データセット生成
            self.monitor.log_event("generation", "Generating dataset")
            generation_start = datetime.now()
            
            # 全ドキュメントをフラット化
            all_docs_flat = []
            for documents in all_documents:
                all_docs_flat.extend(documents)
            
            # データセット生成（ここでは例としてLlamaIndexDatasetFactoryを使用）
            dataset_factory = LlamaIndexDatasetFactory()
            num_questions = 10  # デフォルト値、設定から取得することも可能
            
            dataset = dataset_factory.create_dataset_from_document(
                documents=all_docs_flat,
                num_questions=num_questions
            )
            
            generation_duration = (datetime.now() - generation_start).total_seconds()
            
            self.monitor.log_event(
                "generation",
                "Dataset generation completed",
                {
                    "duration": generation_duration,
                    "total_documents": len(all_docs_flat),
                    "total_questions": len(dataset) if dataset else 0
                }
            )
            
            # メトリクス更新
            self.monitor.update_metrics(
                total_documents=len(all_docs_flat)
            )
            
            # 結果データ
            result_data = {
                "total_documents": len(all_docs_flat),
                "total_questions": len(dataset) if dataset else 0,
                "pattern": pattern_name,
                "config": pattern,
                "template_prompts": template_info
            }
            
            self.monitor.end_test(True, result_data)
            
            return DatasetGenerateTestResult(
                success=True,
                message="Dataset generation completed successfully.",
                data=result_data,
                experiment_id=experiment_id,
                start_time=self.monitor.start_time.isoformat() if self.monitor.start_time else None,
                duration_seconds=generation_duration
            )
            
        except Exception as e:
            logger.error(f"Experiment error: {e}", exc_info=True)
            self.monitor.log_event("error", str(e))
            self.monitor.end_test(False, {"error": str(e)})
            
            return DatasetGenerateTestResult(
                success=False,
                message=str(e),
                data={},
                experiment_id=experiment_id
            )
