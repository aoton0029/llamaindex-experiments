import logging
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import os
from pathlib import Path
import pandas as pd
from llama_index.core import Settings, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import BaseLLM
from llama_index.core.indices.base import BaseIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.schema import Document
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
    RagasDatasetFactory,
    RagasEvaluatorFactory,
    LlamaIndexDatasetFactory,
    LlamaIndexEvaluatorFactory,
    ResponseSynthesizerFactory,
    QueryEngineFactory
)
from transformers import AutoTokenizer
from llama_index.core import load_indices_from_storage, load_index_from_storage

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
        exp_patterns = self.config_manager.get_experiment_pattern(pattern_name)
        tokenizer_config = self.config_manager.get_tokenizer_config_pattern(pattern_name)
        chunking_config = self.config_manager.get_chunking_config_pattern(pattern_name)
        llm_config = self.config_manager.get_llm_config_pattern(pattern_name)
        embedding_config = self.config_manager.get_embedding_config_pattern(pattern_name)
        
        
        # 実験開始
        experiment_id = self.monitor.start_experiment(pattern_name, exp_patterns)
        
        try:
            # トークナイザーの設定
            self.monitor.log_event("setup", "Setting up tokenizer")
            Settings.tokenizer = self._setup_tokenizer(tokenizer_config)
            
            # LLMの設定
            self.monitor.log_event("setup", "Setting up LLM")
            Settings.llm = self._setup_llm(llm_config)
            
            # 埋め込みモデルの設定
            self.monitor.log_event("setup", "Setting up embedding model")
            Settings.embed_model, dim = self._setup_embedding(embedding_config)
            
            # スキーマの設定
            self.monitor.log_event("setup", "Building schema")
            vector_store_collection_schema = SchemaBuilder.build_schema(self.config_manager.get_schema_config('vector_store_schema'), dim)
            image_store_collection_schema = SchemaBuilder.build_schema(self.config_manager.get_schema_config("image_store_schema"), dim)
            
            # データベースの設定と初期化
            self.monitor.log_event("setup", "Initializing database")
            db_manager = DatabaseManager()
            db_manager.initialize(
                milvus_vector_collection_name=f"{pattern_name}_vectore",
                milvus_image_collection_name=f"{pattern_name}_image",
                milvus_image_dim=dim,
                milvus_vector_dim=dim,
                milvus_vector_collection_schema=vector_store_collection_schema,
                milvus_image_collection_schema=image_store_collection_schema,
                mongodb_namespace=pattern_name,
                index_namespace=pattern_name,
                recreate_collection=True
            )
            
            # ストレージコンテキストの取得
            storage_context = db_manager.get_storage_context(
                docstore_namespace=f"{pattern_name}_vectore",
                index_namespace=f"{pattern_name}_image",
                vector_collection=pattern_name,
                image_collection=pattern_name
            )
            
            # ドキュメントのロードとチャンク化
            self.monitor.log_event("processing", "Loading documents")
            document_loader = DocumentLoader()
            all_documents = document_loader.load_from_directory(self.data_dir)
            
            total_nodes = 0
            total_docs = 0
            all_docs_flat = []
            indices = []

            # Step 1: ドキュメントごとにインデックスを作成（同一StorageContextを使用）
            self.monitor.log_event("indexing", "Building indices from all documents")
            indexing_start = datetime.now()
            
            for idx, documents in enumerate(all_documents):
                self.monitor.log_event("indexing", f"Processing document set {idx+1}/{len(all_documents)}")

                # チャンク化を実行
                chunker = self._setup_chunker(chunking_config)
                nodes = chunker.chunk_documents(documents)
                
                # インデックスを構築（同一StorageContextに追加）
                index_builder = self._setup_indexbuilder(
                    exp_patterns['indexing_config'],
                    storage_context=storage_context
                )
                index = index_builder.build_from_nodes(nodes)
                indices.append(index)

                total_nodes += len(nodes)
                total_docs += len(documents)
                all_docs_flat.extend(documents)
            
            indexing_duration = (datetime.now() - indexing_start).total_seconds()
            self.monitor.log_event(
                "indexing",
                "All indices built successfully",
                {
                    "duration": indexing_duration,
                    "total_indices": len(indices),
                    "total_documents": total_docs,
                    "total_nodes": total_nodes
                }
            )
            self.monitor.update_metrics(
                indexing_time=indexing_duration,
                total_documents=total_docs
            )
            
            # Step 2: 統合評価
            integrated_eval_result = None
            if indices:
                self.monitor.log_event("evaluation", "Starting integrated evaluation with all indices")
                integrated_eval_result = self._evaluate_index(
                    exp_patterns['evaluation_config'],
                    index=indices[-1],
                    documents=all_docs_flat,
                )
                self.monitor.log_event(
                    "evaluation",
                    "Integrated evaluation completed",
                    {
                        "total_documents_evaluated": len(all_docs_flat),
                        "total_indices_used": len(indices)
                    }
                )

            # 成功時の結果データに評価結果を追加
            result_data = {
                "total_documents": total_docs,
                "total_nodes": total_nodes,
                "total_indices": len(indices),
                "pattern": pattern_name,
                "config": exp_patterns,
                "evaluation_results": integrated_eval_result
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


    def _setup_llm(self, llm_config: Dict[str, Any]):
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
        try:
            self.monitor.log_event("setup", "Setting up embedding model...")
            backend = embedding_config["backend"]
            model_name = embedding_config["model_name"]
            base_url = embedding_config["base?url"]
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
        try:
            self.monitor.log_event("setup", "Setting up tokenizer...")
            model_name = tokenizer_config["model_name"]
            tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config['kwargs'])
            self.monitor.log_event("setup", f"Set up tokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Tokenizer setup failed: {e}")
            raise

    def _setup_indexbuilder(self, pattern_config: Dict[str, Any], storage_context: StorageContext) -> IndexBuilder:
        try:
            self.monitor.log_event("setup", "Setting up index builder...")
            builder_type = pattern_config.get("type")
            index_builder = IndexBuilderFactory.create(
                builder_type=builder_type,
                storage_context=storage_context,
                show_progress=True, 
                **pattern_config.get("kwargs", {})
            )
            self.monitor.log_event("setup", f"Set up index builder: {builder_type}")
            return index_builder
        except Exception as e:
            logger.error(f": {e}")
            raise

    def _setup_chunker(self, chunking_config: Dict[str, Any]) -> BaseChunker:
        try:
            self.monitor.log_event("setup", "Setting up chunker...")
            chunker_type = chunking_config['type']
            chunker = ChunkerFactory.create(
                chunker_type=chunker_type,
                **chunking_config.get("kwargs", {})
            )
            self.monitor.log_event("setup", f"Set up chunker: {chunker_type}")
            return chunker
        except Exception as e:
            logger.error(f"Chunker setup failed: {e}")
            raise
    
    
    def _evaluate_index(
        self,
        pattern_config: Dict[str, Any],
        index: BaseIndex,
        documents: List[Document],
    ) -> Optional[Dict[str, Any]]:
        """
        インデックスの評価を実行
        
        Args:
            pattern_config: 評価設定
            index: 評価対象のインデックス
            documents: 元のドキュメント
        
        Returns:
            評価結果の辞書
        """
        try:
            self.monitor.log_event("evaluation", "Starting index evaluation")
            
            # 評価設定を取得
            eval_type = pattern_config.get("type")
            eval_config = self.config_manager.get_evaluation_config(eval_type)
            framework = eval_config.get("framework", "llamaindex")
            
            # データセットの準備
            dataset_config = eval_config.get("dataset", {})
            dataset_source = dataset_config.get("source", "auto")
            
            self.monitor.log_event("evaluation", f"Dataset source: {dataset_source}")
            
            # データセットの取得
            if dataset_source == "file":
                # ファイルからデータセットを読み込み
                file_path = dataset_config.get("file_path")
                if not file_path:
                    raise ValueError("dataset.file_path is required when source is 'file'")
                
                self.monitor.log_event("evaluation", f"Loading dataset from file: {file_path}")
                
                if framework == "llamaindex":
                    dataset_factory = LlamaIndexDatasetFactory()
                    dataset = dataset_factory.load_dataset_from_json(file_path)
                elif framework == "ragas":
                    dataset_factory = RagasDatasetFactory()
                    dataset = dataset_factory.load_dataset_from_json(file_path)
                else:
                    raise ValueError(f"Unknown framework: {framework}")
                    
            elif dataset_source == "auto":
                # ドキュメントから自動生成
                num_questions = dataset_config.get("num_questions", 10)
                self.monitor.log_event("evaluation", f"Generating dataset with {num_questions} questions")
                
                if framework == "llamaindex":
                    dataset_factory = LlamaIndexDatasetFactory()
                    dataset = dataset_factory.create_dataset_from_document(
                        documents=documents,
                        num_questions=num_questions,
                        **dataset_config
                    )
                elif framework == "ragas":
                    dataset_factory = RagasDatasetFactory()
                    dataset = dataset_factory.create_dataset_from_document(
                        documents=documents,
                        num_questions=num_questions,
                        **dataset_config
                    )
                else:
                    raise ValueError(f"Unknown framework: {framework}")
            else:
                raise ValueError(f"Unknown dataset source: {dataset_source}")
            
            # クエリエンジンの構築
            self.monitor.log_event("evaluation", "Building query engine")
            top_k = eval_config.get("top_k", 3)
            
            # ResponseSynthesizerの設定
            response_synthesizer = ResponseSynthesizerFactory.get(
                response_mode="compact"
)
            
            # QueryEngineの構築
            query_engine = QueryEngineFactory.create(
                index=index,
                response_synthesizer=response_synthesizer,
                similarity_top_k=top_k
            )
            
            # 評価の実行
            self.monitor.log_event("evaluation", f"Running {framework} evaluation")
            evaluator_types = eval_config.get("evaluators", [])
            
            if framework == "llamaindex":
                evaluator_factory = LlamaIndexEvaluatorFactory()
                results = evaluator_factory.evaluate(
                    dataset=dataset,
                    query_engine=query_engine,
                    evaluator_types=evaluator_types
                )
            elif framework == "ragas":
                evaluator_factory = RagasEvaluatorFactory()
                results = evaluator_factory.evaluate(
                    dataset=dataset,
                    query_engine=query_engine,
                    evaluator_types=evaluator_types
                )
            else:
                raise ValueError(f"Unknown framework: {framework}")
            
            self.monitor.log_event("evaluation", "Evaluation completed successfully")
            
            return {
                "framework": framework,
                "evaluator_types": evaluator_types,
                "dataset_size": len(dataset),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            self.monitor.log_event("evaluation", f"Evaluation error: {str(e)}")
            return None
