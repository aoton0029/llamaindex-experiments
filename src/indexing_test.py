import os
import sys
import json
import logging
from pprint import pprint
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel

from llama_index.core import Settings
from llama_index.core.schema import BaseNode
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from factories.template_prompts import TemplatePromptSettings
from config_manager import ConfigManager
from test_pattern_manager import TestPatternManager
from test_runner_base import TestRunnerBase
from models import (
    GlossaryTerm,
    TechColumnTerm,
    PdfDocumentVector,
)
from factories import (
    DocumentLoader,
)

logger = logging.getLogger(__name__)

llamadebughandler = LlamaDebugHandler()
callback_manager = CallbackManager([llamadebughandler])
Settings.callback_manager = callback_manager

class IndexingTestResult(BaseModel):
    """インデクシングテスト結果"""
    success: bool
    message: str
    data: Dict[str, Any]
    test_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None


class IndexingTestRunner(TestRunnerBase):
    def run_pattern(self, pattern_name: str) -> IndexingTestResult:
        """
        指定されたパターン名でインデクシングテストを実行
        
        Args:
            pattern_name: テストパターン名（例: "test_basic_vector"）
            
        Returns:
            IndexingTestResult: テスト結果
        """
        start_time = datetime.now()
        test_id = f"{pattern_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.monitor.log_event("test_start", f"Starting test pattern: {pattern_name}")
            
            # テストパターンの読み込み
            pattern_config = self.test_pattern_manager.get_test_pattern(
                "indexing_test_patterns", 
                pattern_name
            )
            
            if not pattern_config.get("enabled", False):
                return IndexingTestResult(
                    success=False,
                    message=f"Test pattern '{pattern_name}' is disabled",
                    data={},
                    test_id=test_id,
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration_seconds=0
                )
            
            self.monitor.log_event("config", f"Test name: {pattern_config.get('name')}")
            self.monitor.log_event("config", f"Description: {pattern_config.get('description')}")
            
            # DatabaseManager と StorageContextManager のセットアップ
            db_manager = self._setup_database_manager()
            storage_context_manager = self._setup_storage_context_manager(db_manager)
            
            # StorageContext のセットアップ
            storage_config_dict = pattern_config.get("storage_config", {})
            embedding_config_model = pattern_config.get("embedding_config_model")
            
            # 埋め込みモデルの次元数を取得してストレージ設定に反映
            _, dim = self._setup_embedding(embedding_config_model)
            if "vector_store" in storage_config_dict:
                storage_config_dict["vector_store"]["dim"] = dim
            
            storage_context = self._setup_storage_context(
                storage_config_dict, 
                storage_context_manager
            )
            
            # LLM のセットアップ
            llm_config_model = pattern_config.get("llm_config_model")
            llm = self._setup_llm(llm_config_model)
            Settings.llm = llm
            
            # Embedding のセットアップ
            embedding, _ = self._setup_embedding(embedding_config_model)
            Settings.embed_model = embedding
            
            # Tokenizer のセットアップ（オプション）
            tokenizer_config_model = pattern_config.get("tokenizer_config_model")
            if tokenizer_config_model:
                tokenizer = self._setup_tokenizer(tokenizer_config_model)
                Settings.tokenizer = tokenizer
            
            # テンプレートプロンプトの初期化
            TemplatePromptSettings.initialize(self.config_manager.get_template_prompts())
            
            # データソースの読み込み
            data_source = pattern_config.get("data_source", {})
            documents = self._load_documents(data_source)
            self.monitor.log_event("data_load", f"Loaded {len(documents)} documents")
            
            # インデクシングパターンの実行
            indexing_patterns = pattern_config.get("indexing_pattern", [])
            indexing_results = []
            
            for idx, indexing_config in enumerate(indexing_patterns):
                self.monitor.log_event("indexing", f"Processing pattern {idx + 1}/{len(indexing_patterns)}")
                
                index_type = indexing_config.get("type")
                chunking_config_model = indexing_config.get("chunking_config_model")
                extractor_pattern = indexing_config.get("extractor_pattern")
                
                # チャンカーのセットアップ
                chunker = self._setup_chunker(chunking_config_model)
                
                # エクストラクタのセットアップ
                extractors = []
                if extractor_pattern:
                    extractors = self._setup_extractors(extractor_pattern)
                
                # インデックスビルダーのセットアップ
                index_builder = self._setup_indexbuilder(index_type, storage_context)
                
                # インデックスの構築
                self.monitor.log_event("indexing", f"Building {index_type} index...")
                
                # ドキュメントをチャンキング
                nodes = chunker.get_nodes_from_documents(documents, show_progress=True)
                self.monitor.log_event("chunking", f"Created {len(nodes)} nodes from chunking")
                
                # メタデータ抽出の実行
                if extractors:
                    self.monitor.log_event("extraction", f"Running {len(extractors)} extractors...")
                    for extractor in extractors:
                        nodes = extractor.process_nodes(nodes, show_progress=True)
                    self.monitor.log_event("extraction", "Metadata extraction completed")
                
                # インデックスの作成
                index = index_builder.build_from_nodes(nodes)
                self.monitor.log_event("indexing", f"Index built successfully with {len(nodes)} nodes")

                indexing_results.append({
                    "index_type": index_type,
                    "chunking_config": chunking_config_model,
                    "extractor_pattern": extractor_pattern,
                    "node_count": len(nodes),
                    "document_count": len(documents)
                })
                
                self.monitor.log_event("indexing", f"Created {len(nodes)} nodes")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = IndexingTestResult(
                success=True,
                message=f"Test pattern '{pattern_name}' completed successfully",
                data={
                    "pattern_name": pattern_name,
                    "pattern_config": pattern_config,
                    "indexing_results": indexing_results,
                    "storage_context_name": storage_config_dict.get("context_name")
                },
                test_id=test_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration
            )
            
            self.monitor.log_event("test_end", f"Test completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Test pattern '{pattern_name}' failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.monitor.log_event("error", error_msg)
            
            return IndexingTestResult(
                success=False,
                message=error_msg,
                data={"error": str(e)},
                test_id=test_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration
            )
    
    def _load_documents(self, data_source: Dict[str, Any]) -> List:
        """
        データソースからドキュメントを読み込む
        
        Args:
            data_source: データソース設定
            
        Returns:
            読み込まれたドキュメントのリスト
        """
        source_type = data_source.get("type", "pdf")
        directory = data_source.get("directory")
        file_pattern = data_source.get("file_pattern", "*.pdf")
        
        loader = DocumentLoader()
        
        if source_type == "pdf":
            documents = loader.load_from_directory(
                directory=directory,
                file_pattern=file_pattern
            )
        elif source_type == "markdown":
            documents = loader.load_from_directory(
                directory=directory,
                file_pattern=file_pattern
            )
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
        
        return documents
    
    def cleanup_test_context(self, pattern_name: str):
        """
        テストで使用したStorageContextをクリーンアップ
        
        Args:
            pattern_name: テストパターン名
        """
        try:
            pattern_config = self.test_pattern_manager.get_test_pattern(
                "indexing_test_patterns", 
                pattern_name
            )
            storage_config = pattern_config.get("storage_config", {})
            context_name = storage_config.get("context_name")
            
            if context_name:
                self._drop_storage_context(context_name)
        except Exception as e:
            logger.error(f"Failed to cleanup test context: {e}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run indexing tests")
    parser.add_argument(
        "--pattern",
        type=str,
        default="test_basic_vector",
        help="Test pattern name to run"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Cleanup test context after execution"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="/workspace/src/config",
        help="Configuration directory path"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/workspace/src/tests",
        help="Test patterns directory path"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="/workspace/results",
        help="Results output directory path"
    )
    
    args = parser.parse_args()
    
    # テストランナーの初期化
    runner = IndexingTestRunner(
        config_dir=args.config_dir,
        test_dir=args.test_dir,
        result_dir=args.result_dir
    )
    
    # テストの実行
    result = runner.run_pattern(args.pattern)
    
    # 結果の表示
    print("\n" + "="*80)
    print("TEST RESULT")
    print("="*80)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Test ID: {result.test_id}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print("\nData:")
    pprint(result.data)
    
    # 結果の保存
    result_file = Path(args.result_dir) / f"{result.test_id}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"\nResult saved to: {result_file}")
    
    # クリーンアップ
    if args.cleanup:
        print("\nCleaning up test context...")
        runner.cleanup_test_context(args.pattern)
        print("Cleanup completed")
    
    # 終了コード
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
