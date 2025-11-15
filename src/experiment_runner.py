import logging
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path
import pandas as pd
from llama_index.core import Settings, StorageContext
from llama_index.core.indices.base import BaseIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from db.database_manager import DatabaseConfig, DatabaseManager
from services import ConfigManager
from experiment_monitor import ExperimentMonitor, ExperimentMetrics
from factories import (
    LLMFactory, 
    EmbeddingFactory, 
    DocumentLoader, 
    ChunkerFactory, 
    IndexBuilderFactory, 
    BaseChunker, 
    SchemaBuilder,
    IndexBuilder,
    , 
    EvaluatorFactory
)
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

llamadebughandler = LlamaDebugHandler()
callback_manager = CallbackManager([llamadebughandler])
Settings.callback_manager = callback_manager

class ExperimentResult(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]
    experiment_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

class ExperimentRunner:
    def __init__(self, config_dir: str, data_dir: str, result_dir: str):
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.config_manager = ConfigManager(config_dir)
        self.monitor = ExperimentMonitor(result_dir)

    def run_all_experiments(self):
        self.config_manager.load_all_configs()
        experiments: Dict[str, Any] = self.config_manager.get_config("test_patterns")
        results: List[ExperimentResult] = []
        
        for experiment_name, experiment_config in experiments.items():
            if not experiment_config.get("enabled", True):
                logger.info(f"Skipping disabled experiment: {experiment_name}")
                continue
            try:
                result = self.run_experiment(experiment_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                results.append(ExperimentResult(
                    success=False, 
                    message=str(e), 
                    data={},
                    experiment_id=experiment_name
                ))
        
        # 全実験の統合サマリーを作成
        self._create_all_experiments_summary(results)
        
        return results

    def run_experiment(self, pattern_name: str):
        # 指定された実験パターンを実行
        exp_patterns = self.config_manager.get_experiment_patterns(pattern_name)
        
        # 実験開始
        experiment_id = self.monitor.start_experiment(pattern_name, exp_patterns)
        
        try:
            # トークナイザーの設定
            self.monitor.log_event("setup", "Setting up tokenizer")
            Settings.tokenizer = self._setup_tokenizer(exp_patterns['tokenizer_config'])
            
            # LLMの設定
            self.monitor.log_event("setup", "Setting up LLM")
            Settings.llm = self._setup_llm(exp_patterns['llm_config'])
            
            # 埋め込みモデルの設定
            self.monitor.log_event("setup", "Setting up embedding model")
            Settings.embed_model = self._setup_embedding(exp_patterns['embedding_config'])
            
            # スキーマの設定
            self.monitor.log_event("setup", "Building schema")
            dim = exp_patterns['embedding_config']['dimensions']
            collection_schema = SchemaBuilder.build_schema(
                self.config_manager.get_schema_config('schema'), 
                dim
            )
            
            # データベースの設定と初期化
            self.monitor.log_event("setup", "Initializing database")
            db_config = DatabaseConfig(
                milvus_dim=dim,
                milvus_collection_schema=collection_schema
            )
            db_manager = DatabaseManager(db_config)
            db_manager.initialize(True)
            
            # ストレージコンテキストの取得
            storage_context = db_manager.get_storage_context(**exp_patterns['storage'])
            
            # ドキュメントのロードとチャンク化
            self.monitor.log_event("processing", "Loading documents")
            document_loader = DocumentLoader()
            all_documents = document_loader.load_from_directory(self.data_dir)
            
            total_nodes = 0
            total_docs = 0

            indices: List[BaseIndex] = []
            all_eval_results = []
            for documents in all_documents:
                self.monitor.log_event("indexing", "Building index")
                indexing_start = datetime.now()

                # チャンク化を実行
                chunker = self._setup_chunker(exp_patterns['chunker_config'])
                nodes = chunker.chunk_documents(documents)
                
                # インデックスを構築
                index_builder = self._setup_indexbuilder(
                    exp_patterns['indexing_config'],
                    storage_context=storage_context
                )
                index = index_builder.build_from_nodes(nodes)
                indices.append(index)

                total_nodes += len(nodes)
                total_docs += len(documents)
                indexing_duration = (datetime.now() - indexing_start).total_seconds()
                self.monitor.log_event(
                    "indexing", 
                    "Index built successfully",
                    {"duration": indexing_duration}
                )
                self.monitor.update_metrics(
                    indexing_time=indexing_duration,
                    total_documents=total_docs
                )
                
                # 評価を実行
                eval_result = self._evaluate_index(
                    index=index,
                    documents=documents,
                    experiment_dir=self.monitor.get_experiment_dir()
                )
                all_eval_results.append(eval_result)

            # 成功時の結果データに評価結果を追加
            result_data = {
                "total_documents": total_docs,
                "total_nodes": total_nodes,
                "pattern": pattern_name,
                "config": exp_patterns,
                "evaluation_results": all_eval_results
            }
            
            self.monitor.end_experiment(True, result_data)
            
            return ExperimentResult(
                success=True, 
                message="Experiment completed successfully.",
                data=result_data,
                experiment_id=experiment_id,
                start_time=self.monitor.start_time.isoformat()
            )
            
        except Exception as e:
            logger.error(f"Experiment error: {e}", exc_info=True)
            self.monitor.log_event("error", str(e))
            self.monitor.end_experiment(False, {"error": str(e)})
            
            return ExperimentResult(
                success=False,
                message=str(e),
                data={},
                experiment_id=experiment_id
            )
    
    def _create_all_experiments_summary(self, results: List[ExperimentResult]):
        """全実験の統合サマリーを作成"""
        summary_path = Path(self.result_dir) / "all_experiments_summary.json"
        
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
        df = pd.DataFrame([
            {
                "experiment_id": r.experiment_id,
                "success": r.success,
                "message": r.message,
                "duration_seconds": r.duration_seconds
            }
            for r in results
        ])
        csv_path = Path(self.result_dir) / "all_experiments_summary.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _setup_llm(self, config: Dict[str, Any]):
        backend = config["backend"]
        model_name = config["model_name"]
        
        llm = LLMFactory.create(
            backend=backend,
            model_name=model_name,
            **config.get("kwargs", {})
        )
        return llm

    def _setup_embedding(self, config: Dict[str, Any]):
        backend = config["backend"]
        model_name = config["model_name"]

        embedding = EmbeddingFactory.create(
            backend=backend,
            model_name=model_name,
            **config.get("kwargs", {})
        )
        return embedding

    def _setup_tokenizer(self, config: Dict[str, Any]):
        tokenizer_type = config.get("type")
        tokenizer = AutoTokenizer.from_pretrained("")
        return tokenizer

    def _setup_indexbuilder(self, config: Dict[str, Any], storage_context: StorageContext) -> IndexBuilder:
        index_builder = IndexBuilderFactory.create(
            builder_type=config.get("type"),
            storage_context=storage_context,
            show_progress=True, 
            **config.get("kwargs", {})
        )
        return index_builder

    def _setup_chunker(self, config: Dict[str, Any]) -> BaseChunker:
        chunker = ChunkerFactory.create(
            backend=config["backend"],
            model_name=config["model_name"],
            **config.get("kwargs", {})
        )
        return chunker
    
    def _evaluate_index(self, index: BaseIndex, documents: List, experiment_dir: Path) -> Dict[str, Any]:
        """
        インデックスを評価
        
        Args:
            index: 評価対象のインデックス
            documents: 元のドキュメント
            experiment_dir: 実験結果の保存先ディレクトリ
            
        Returns:
            評価結果の辞書
        """
        try:
            # 評価設定を取得
            exp_patterns = self.config_manager.get_experiment_patterns(self.monitor.current_experiment_name)
            eval_config = exp_patterns.get('evaluation_config', {})
            
            if not eval_config.get('enabled', True):
                logger.info("Evaluation is disabled")
                return {"success": True, "message": "Evaluation skipped"}
            
            # 評価エンジンの選択（デフォルトはllamaindex）
            eval_engine = eval_config.get('engine', 'llamaindex')
            
            if eval_engine == 'ragas':
                # return self._evaluate_with_ragas(index, documents, experiment_dir, eval_config)
                return {"success": False, "message": "Ragas evaluation not implemented yet"}
            elif eval_engine == 'llamaindex':
                return self._evaluate_with_llamaindex(index, documents, experiment_dir, eval_config)
            else:
                raise ValueError(f"Unknown evaluation engine: {eval_engine}")
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            self.monitor.log_event("evaluation", f"Evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _evaluate_with_llamaindex(
        self, 
        index: BaseIndex, 
        documents: List, 
        experiment_dir: Path,
        eval_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LlamaIndexを使った評価"""
        try:
            self.monitor.log_event("evaluation", "Starting LlamaIndex evaluation")
            
            # データセット生成設定を取得
            dataset_config = eval_config.get('dataset', {})
            testset_size = dataset_config.get('size', eval_config.get('testset_size', 10))
            dataset_source = dataset_config.get('source', 'auto')  # 'auto', 'file', 'huggingface'
            
            # データセットの取得または生成
            if dataset_source == 'file':
                # ファイルから読み込み
                dataset_path = dataset_config.get('path')
                if not dataset_path:
                    raise ValueError("Dataset path is required when source is 'file'")
                
                self.monitor.log_event("evaluation", f"Loading dataset from file: {dataset_path}")
                dataset = LlamaIndexDatasetFactory.load_dataset_from_file(
                    file_path=dataset_path,
                    format=dataset_config.get('format', 'json')
                )
                
            elif dataset_source == 'huggingface':
                # HuggingFaceから読み込み
                dataset_name = dataset_config.get('name')
                if not dataset_name:
                    raise ValueError("Dataset name is required when source is 'huggingface'")
                
                self.monitor.log_event("evaluation", f"Loading dataset from HuggingFace: {dataset_name}")
                dataset = LlamaIndexDatasetFactory.load_huggingface_dataset(
                    dataset_name=dataset_name,
                    split=dataset_config.get('split', 'train'),
                    query_field=dataset_config.get('query_field', 'query'),
                    answer_field=dataset_config.get('answer_field', 'answer'),
                    context_field=dataset_config.get('context_field')
                )
                
            else:  # 'auto' - 自動生成
                self.monitor.log_event("evaluation", f"Auto-generating dataset with {testset_size} questions")
                
                # データセット保存パスの設定
                dataset_save_path = experiment_dir / "evaluation" / "generated_dataset.json"
                dataset_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # データセット自動生成
                dataset = LlamaIndexDatasetFactory.auto_generate_dataset(
                    documents=documents,
                    num_questions=testset_size,
                    num_questions_per_chunk=dataset_config.get('questions_per_chunk', 2),
                    save_path=str(dataset_save_path),
                    include_contexts=dataset_config.get('include_contexts', True),
                    question_types=dataset_config.get('question_types')
                )
                
                self.monitor.log_event(
                    "evaluation", 
                    f"Dataset generated and saved to {dataset_save_path}"
                )
            
            # 評価器の設定
            evaluator_types = eval_config.get('evaluators', ['faithfulness', 'relevancy'])
            evaluators = []
            
            for eval_type in evaluator_types:
                self.monitor.log_event("evaluation", f"Creating evaluator: {eval_type}")
                
                # 評価器固有の設定を取得
                evaluator_config = eval_config.get(f'{eval_type}_config', {})
                
                if eval_type == 'semantic_similarity':
                    # 日本語対応のsemantic_similarity評価器
                    evaluator = EvaluatorFactory.create_semantic_similarity_evaluator(
                        similarity_threshold=evaluator_config.get('threshold', 0.8),
                        use_japanese_model=evaluator_config.get('use_japanese_model', True),
                        embed_model=Settings.embed_model if evaluator_config.get('use_global_embed', True) else None
                    )
                elif eval_type == 'correctness':
                    evaluator = EvaluatorFactory.create_correctness_evaluator(
                        score_threshold=evaluator_config.get('threshold', 3.0)
                    )
                else:
                    evaluator = EvaluatorFactory.create_evaluator(eval_type, **evaluator_config)
                
                if evaluator:
                    evaluators.append(evaluator)
            
            if not evaluators:
                return {
                    "success": False,
                    "error": "No valid evaluators were created"
                }
            
            # クエリエンジンの作成
            self.monitor.log_event("evaluation", "Creating query engine")
            query_engine = index.as_query_engine(
                similarity_top_k=eval_config.get('top_k', 3)
            )
            
            # 評価の実行
            self.monitor.log_event("evaluation", f"Running evaluation with {len(evaluators)} evaluators")
            eval_results = {}
            
            from llama_index.core.evaluation import BatchEvalRunner
            
            # QueryResponseDatasetから予測を生成
            predictions = {}
            for query_id, query in dataset.queries.items():
                response = query_engine.query(query)
                predictions[query_id] = str(response)
            
            # 各評価器で評価
            for evaluator in evaluators:
                evaluator_name = evaluator.__class__.__name__
                self.monitor.log_event("evaluation", f"Running {evaluator_name}")
                
                eval_runner = BatchEvalRunner(
                    evaluators={evaluator_name: evaluator},
                    show_progress=True
                )
                
                # 評価実行
                results = eval_runner.evaluate_responses(
                    queries=list(dataset.queries.values()),
                    responses=list(predictions.values()),
                    reference=list(dataset.responses.values()) if dataset.responses else None,
                    contexts=[dataset.reference_contexts.get(qid, []) for qid in dataset.queries.keys()] if dataset.reference_contexts else None
                )
                
                eval_results[evaluator_name] = results
            
            # 結果の集約
            aggregated_results = self._aggregate_evaluation_results(eval_results)
            
            # 結果の保存
            results_path = experiment_dir / "evaluation" / "evaluation_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_results, f, ensure_ascii=False, indent=2)
            
            self.monitor.log_event("evaluation", f"Evaluation completed. Results saved to {results_path}")
            
            return {
                "success": True,
                "results": aggregated_results,
                "dataset_size": len(dataset.queries),
                "evaluators": [e.__class__.__name__ for e in evaluators]
            }
            
        except Exception as e:
            logger.error(f"LlamaIndex evaluation error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _aggregate_evaluation_results(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """評価結果を集約"""
        aggregated = {
            "summary": {},
            "details": eval_results
        }
        
        # 各評価器の平均スコアを計算
        for evaluator_name, results in eval_results.items():
            if hasattr(results, 'scores'):
                scores = [r for r in results.scores if r is not None]
                if scores:
                    aggregated["summary"][evaluator_name] = {
                        "mean": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores),
                        "count": len(scores)
                    }
        
        return aggregated
