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
from llama_index.core.ingestion import IngestionPipeline

from transformers import AutoTokenizer
from factories.document_loader import DocumentLoader
from factories.template_prompts import TemplatePromptSettings
from services.config_manager import ConfigManager
from test_runner_base import TestRunnerBase

logger = logging.getLogger(__name__)

llamadebughandler = LlamaDebugHandler()
callback_manager = CallbackManager([llamadebughandler])
Settings.callback_manager = callback_manager


class ExtractorTestResult(BaseModel):
    """エクストラクタテスト結果"""
    success: bool
    message: str
    data: Dict[str, Any]
    experiment_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None


class ExtractorTestRunner(TestRunnerBase):
    """エクストラクタテストを実行するクラス"""
    
    def run_all_tests(self) -> List[ExtractorTestResult]:
        """全てのエクストラクタテスト実験を実行"""
        self.config_manager.load_all_configs()
        
        # テンプレートプロンプトを初期化
        TemplatePromptSettings.initialize(self.config_manager)
        TemplatePromptSettings._load_templates()
        
        experiments: Dict[str, Any] = self.config_manager.get_extractor_test_patterns()
        results: List[ExtractorTestResult] = []
        
        for experiment_name, experiment_config in experiments.items():
            if not experiment_config.get("enabled", True):
                logger.info(f"Skipping disabled experiment: {experiment_name}")
                continue
            try:
                result = self.run_test(experiment_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                results.append(ExtractorTestResult(
                    success=False,
                    message=str(e),
                    data={},
                    experiment_id=experiment_name
                ))
        
        # 全実験の統合サマリーを作成
        self._create_all_tests_summary(results)
        
        return results
    
    def run_test(self, pattern_name: str) -> ExtractorTestResult:
        """指定されたパターンでエクストラクタテスト実験を実行"""
        
        # テストパターンを取得
        pattern = self.config_manager.get_test_pattern(pattern_name, "extractor")
        
        # 実験開始
        experiment_id = self.monitor.start_test(pattern_name, pattern)
        experiment_dir = self.monitor.get_test_dir()
        
        try:
            # フェーズ0: テンプレートプロンプト情報を記録
            self.monitor.log_event("setup", "Phase 0: Recording template prompts")
            template_info = TemplatePromptSettings.get_templates_info()
            self._save_phase_result(experiment_dir, "phase0_template_prompts", template_info)
            
            # フェーズ1: トークナイザーの設定
            self.monitor.log_event("setup", "Phase 1: Setting up tokenizer")
            tokenizer_config = self.config_manager.get_tokenizer_config_from_pattern(pattern_name, "extractor")
            Settings.tokenizer = self._setup_tokenizer(tokenizer_config)
            
            # フェーズ2: LLMの設定
            self.monitor.log_event("setup", "Phase 2: Setting up LLM")
            llm_config = self.config_manager.get_llm_config_from_pattern(pattern_name, "extractor")
            Settings.llm = self._setup_llm(llm_config)
            
            # フェーズ3: 埋め込みモデルの設定
            self.monitor.log_event("setup", "Phase 3: Setting up embedding model")
            embedding_config = self.config_manager.get_embedding_config_from_pattern(pattern_name, "extractor")
            Settings.embed_model, dim = self._setup_embedding(embedding_config)
            
            # フェーズ4: チャンキング設定の取得
            chunking_config = self.config_manager.get_chunking_config_from_pattern(pattern_name, "extractor")
            
            # フェーズ5: ドキュメントのロード
            self.monitor.log_event("processing", "Phase 4: Loading documents")
            loading_start = datetime.now()
            document_loader = DocumentLoader()
            all_documents = document_loader.load_from_directory(self.data_dir)
            loading_duration = (datetime.now() - loading_start).total_seconds()
            
            total_docs = sum(len(docs) for docs in all_documents)
            self.monitor.log_event(
                "processing",
                "Document loading completed",
                {"duration": loading_duration, "total_documents": total_docs}
            )
            
            # ドキュメント情報を保存
            doc_info_before = {
                "total_documents": total_docs,
                "loading_duration": loading_duration,
                "documents": []
            }
            for docs in all_documents:
                for doc in docs:
                    doc_info_before["documents"].append({
                        "doc_id": doc.doc_id,
                        "text_length": len(doc.text),
                        "text": doc.text,
                        "metadata": doc.metadata
                    })
            
            self._save_phase_result(experiment_dir, "phase1_document_loading", doc_info_before)
            
            # フェーズ5.5: 前処理の実行
            self.monitor.log_event("preprocessing", "Phase 4.5: Starting preprocessing")
            preprocessing_start = datetime.now()
            
            # 前処理をセットアップ
            preprocessor = self._setup_preprocessor(use_schema_based=True)
            
            # 前処理を実行
            preprocessed_documents = []
            for documents in all_documents:
                processed = preprocessor.process(documents)
                preprocessed_documents.append(processed)
            
            preprocessing_duration = (datetime.now() - preprocessing_start).total_seconds()
            
            self.monitor.log_event(
                "preprocessing",
                "Preprocessing completed",
                {
                    "duration": preprocessing_duration,
                    "total_documents": sum(len(docs) for docs in preprocessed_documents)
                }
            )
            
            # 前処理後のドキュメント情報を保存
            doc_info_after = {
                "total_documents": sum(len(docs) for docs in preprocessed_documents),
                "preprocessing_duration": preprocessing_duration,
                "documents": []
            }
            for docs in preprocessed_documents:
                for doc in docs:
                    doc_info_after["documents"].append({
                        "doc_id": doc.doc_id,
                        "text_length": len(doc.text),
                        "text_preview": doc.text[:200] if len(doc.text) > 200 else doc.text,
                        "metadata": doc.metadata
                    })
            
            self._save_phase_result(experiment_dir, "phase1_document_loading_after", doc_info_after)
            
            # 前処理の効果を比較
            preprocessing_comparison = {
                "before": {
                    "avg_text_length": sum(d["text_length"] for d in doc_info_before["documents"]) / len(doc_info_before["documents"]) if doc_info_before["documents"] else 0,
                    "total_metadata_keys": sum(len(d["metadata"]) for d in doc_info_before["documents"])
                },
                "after": {
                    "avg_text_length": sum(d["text_length"] for d in doc_info_after["documents"]) / len(doc_info_after["documents"]) if doc_info_after["documents"] else 0,
                    "total_metadata_keys": sum(len(d["metadata"]) for d in doc_info_after["documents"])
                }
            }
            self._save_phase_result(experiment_dir, "phase1_preprocessing_comparison", preprocessing_comparison)
            
            # フェーズ6: チャンキング実行
            self.monitor.log_event("chunking", "Phase 5: Starting chunking")
            chunking_start = datetime.now()
            
            chunker = self._setup_chunker(chunking_config)
            
            all_nodes: List[BaseNode] = []
            for documents in all_documents:
                nodes = chunker.get_nodes_from_documents(documents)
                all_nodes.extend(nodes)
            
            chunking_duration = (datetime.now() - chunking_start).total_seconds()
            
            self.monitor.log_event(
                "chunking",
                "Chunking completed",
                {
                    "duration": chunking_duration,
                    "total_nodes": len(all_nodes)
                }
            )
            
            # チャンク情報を保存
            chunk_info = {
                "total_nodes": len(all_nodes),
                "chunking_duration": chunking_duration,
                "avg_nodes_per_doc": len(all_nodes) / total_docs if total_docs > 0 else 0,
                "chunking_config": chunking_config,
                "nodes": []
            }
            
            for i, node in enumerate(all_nodes):  # 最初の100ノードのみ保存
                chunk_info["nodes"].append({
                    "node_id": node.node_id,
                    "text_length": len(node.text),
                    "text_preview": node.text,
                    "metadata": node.metadata
                })
            
            self._save_phase_result(experiment_dir, "phase2_chunking", chunk_info)
            
            # フェーズ7: エクストラクタの設定と実行
            self.monitor.log_event("extraction", "Phase 6: Starting extraction")
            extraction_start = datetime.now()
            
            # エクストラクタパターンを取得
            extractor_pattern = self.config_manager.get_extractor_pattern_from_test(pattern_name)
            extractor_names = extractor_pattern.get("extractors", [])
            
            # エクストラクタ設定を取得
            extractor_configs = []
            for extractor_name in extractor_names:
                extractor_config = self.config_manager.get_extractor_config(extractor_name)
                if extractor_config:
                    extractor_configs.append(extractor_config)
            
            # エクストラクタをセットアップ
            extractors = self._setup_extractors(extractor_configs)
            
            # IngestionPipelineで抽出を実行
            pipeline = IngestionPipeline(transformations=extractors)
            
            extracted_nodes = pipeline.run(nodes=all_nodes, show_progress=True)
            
            extraction_duration = (datetime.now() - extraction_start).total_seconds()
            
            self.monitor.log_event(
                "extraction",
                "Extraction completed",
                {
                    "duration": extraction_duration,
                    "extractor_count": len(extractors),
                    "extractor_types": [e.__class__.__name__ for e in extractors]
                }
            )
            
            # 抽出情報を保存
            extraction_info = {
                "total_extracted_nodes": len(extracted_nodes),
                "extraction_duration": extraction_duration,
                "extractor_pattern": extractor_pattern,
                "extractor_configs": extractor_configs,
                "extractors_used": [e.__class__.__name__ for e in extractors],
                "extracted_metadata_samples": []
            }
            
            # 抽出されたメタデータのサンプルを保存（最初の10ノード）
            for i, node in enumerate(extracted_nodes):
                sample = {
                    "node_id": node.node_id,
                    "text_preview": node.text,
                    "metadata": node.metadata,
                    "excluded_embed_metadata_keys": node.excluded_embed_metadata_keys,
                    "excluded_llm_metadata_keys": node.excluded_llm_metadata_keys
                }
                extraction_info["extracted_metadata_samples"].append(sample)
            
            self._save_phase_result(experiment_dir, "phase3_extraction", extraction_info)
            
            # メトリクス更新
            self.monitor.update_metrics(
                chunking_time=chunking_duration,
                total_documents=total_docs,
                total_nodes=len(extracted_nodes)
            )
            
            # 結果データ
            result_data = {
                "pattern": pattern_name,
                "config": pattern,
                "template_prompts": template_info,
                "phases": {
                    "loading": {
                        "duration": loading_duration,
                        "total_documents": total_docs
                    },
                    "chunking": {
                        "duration": chunking_duration,
                        "total_nodes": len(all_nodes),
                        "config": chunking_config
                    },
                    "extraction": {
                        "duration": extraction_duration,
                        "total_extracted_nodes": len(extracted_nodes),
                        "extractors_used": [e.__class__.__name__ for e in extractors]
                    }
                },
                "summary": {
                    "total_duration": loading_duration + chunking_duration + extraction_duration,
                    "final_node_count": len(extracted_nodes)
                }
            }
            
            self.monitor.end_test(True, result_data)
            
            return ExtractorTestResult(
                success=True,
                message="Extractor test completed successfully.",
                data=result_data,
                experiment_id=experiment_id,
                start_time=self.monitor.start_time.isoformat() if self.monitor.start_time else None,
                duration_seconds=result_data["summary"]["total_duration"]
            )
            
        except Exception as e:
            logger.error(f"Experiment error: {e}", exc_info=True)
            self.monitor.log_event("error", str(e))
            self.monitor.end_test(False, {"error": str(e)})
            
            return ExtractorTestResult(
                success=False,
                message=str(e),
                data={},
                experiment_id=experiment_id
            )
    
    def _save_phase_result(self, experiment_dir: Path, phase_name: str, data: Dict[str, Any]):
        """各フェーズの結果を保存"""
        phase_dir = experiment_dir / "results" / "phases"
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で保存
        json_path = phase_dir / f"{phase_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved phase result: {phase_name}")
    
    def _create_all_tests_summary(self, results: List[ExtractorTestResult]):
        """全実験の統合サマリーを作成"""
        import pandas as pd
        
        summary_path = Path(self.result_dir) / "all_extractor_experiments_summary.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "experiments": [r.model_dump() for r in results]
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # CSV形式でも保存
        df_data = []
        for r in results:
            row = {
                "experiment_id": r.experiment_id,
                "success": r.success,
                "message": r.message,
                "duration_seconds": r.duration_seconds
            }
            
            # フェーズごとの詳細情報を追加
            if r.data and "phases" in r.data:
                phases = r.data["phases"]
                if "loading" in phases:
                    row["loading_duration"] = phases["loading"]["duration"]
                    row["total_documents"] = phases["loading"]["total_documents"]
                if "chunking" in phases:
                    row["chunking_duration"] = phases["chunking"]["duration"]
                    row["total_nodes"] = phases["chunking"]["total_nodes"]
                if "extraction" in phases:
                    row["extraction_duration"] = phases["extraction"]["duration"]
                    row["extractors_used"] = ", ".join(phases["extraction"]["extractors_used"])
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = Path(self.result_dir) / "all_extractor_experiments_summary.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        logger.info(f"Created summary: {summary_path}")
        logger.info(f"Created CSV summary: {csv_path}")


if __name__ == "__main__":
    # パスの設定
    current_dir = Path(__file__).parent
    config_dir = current_dir / "config"
    data_dir = current_dir.parent / "sample_data" / "pdf"
    result_dir = current_dir / "results" / "extractor_tests"
    
    # テストランナーの実行
    runner = ExtractorTestRunner(
        config_dir=str(config_dir),
        data_dir=str(data_dir),
        result_dir=str(result_dir)
    )
    
    print("=" * 80)
    print("Starting Extractor Test Experiments")
    print("=" * 80)
    
    results = runner.run_all_tests()
    
    print("\n" + "=" * 80)
    print("All Experiments Completed")
    print("=" * 80)
    print(f"Total: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.success)}")
    print(f"Failed: {sum(1 for r in results if not r.success)}")
    print("=" * 80)
