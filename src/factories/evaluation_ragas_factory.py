# from ragas.metrics import (
#     Faithfulness,
#     AnswerRelevancy,
#     ContextPrecision,
#     ContextRecall,
#     ContextRelevance,
#     AnswerSimilarity,
#     AnswerCorrectness,
# )
# from ragas.integrations.llama_index import evaluate
# from ragas.llms import LlamaIndexLLMWrapper
# from ragas.embeddings import LlamaIndexEmbeddingsWrapper
# from llama_index.core import Settings
# from llama_index.core.schema import Document
# from typing import List, Dict, Any, Optional
# from pathlib import Path
# import logging
# import json
# import pandas as pd
# from ragas.dataset import Dataset
# from datasets import load_dataset
# from ragas import EvaluationDataset

# logger = logging.getLogger(__name__)


# class RagasEvaluatorFactory:
#     """
#     Ragas評価機能のファクトリクラス
#     LlamaIndexのSettings経由でLLMと埋め込みモデルを取得して評価を実行
#     オフライン評価のためローカルデータセットをサポート
#     """
    
#     @staticmethod
#     def create_llm_wrapper():
#         """Settings.llmをRagas用にラップ"""
#         if Settings.llm is None:
#             raise ValueError("Settings.llm is not configured")
#         return LlamaIndexLLMWrapper(Settings.llm)
    
#     @staticmethod
#     def create_embeddings_wrapper():
#         """Settings.embed_modelをRagas用にラップ"""
#         if Settings.embed_model is None:
#             raise ValueError("Settings.embed_model is not configured")
#         return LlamaIndexEmbeddingsWrapper(Settings.embed_model)
    
#     @staticmethod
#     def create_evaluation_dataset(name: str, dataset_dir: str, split: List[str] = ["test"]) -> EvaluationDataset:
#         try:
#             eval_dataset = load_dataset(name=name, data_dir=dataset_dir, split=split)
#             eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
#             return eval_dataset
#         except Exception as e:
#             logger.error(f"Failed to load HuggingFace dataset: {e}", exc_info=True)
#             raise
    
#     @staticmethod
#     def create_metrics(
#         metric_names: Optional[List[str]] = None
#     ) -> List[Any]:
#         """
#         評価メトリクスのリストを作成
        
#         Args:
#             metric_names: 使用するメトリクス名のリスト
#                          None の場合はデフォルトメトリクスを使用
                         
#         Returns:
#             メトリクスオブジェクトのリスト
#         """
#         llm = RagasEvaluatorFactory.create_llm_wrapper()
#         embeddings = RagasEvaluatorFactory.create_embeddings_wrapper()
        
#         available_metrics = {
#             "faithfulness": Faithfulness(llm=llm),
#             "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
#             "context_precision": ContextPrecision(llm=llm),
#             "context_recall": ContextRecall(llm=llm),
#             "context_relevance": ContextRelevance(llm=llm),
#             "answer_similarity": AnswerSimilarity(embeddings=embeddings),
#             "answer_correctness": AnswerCorrectness(llm=llm),
#         }
        
#         if metric_names is None:
#             # デフォルトメトリクス
#             metric_names = [
#                 "faithfulness",
#                 "answer_relevancy",
#                 "context_precision",
#                 "context_recall"
#             ]
        
#         metrics = []
#         for name in metric_names:
#             if name in available_metrics:
#                 metrics.append(available_metrics[name])
#                 logger.info(f"Added metric: {name}")
#             else:
#                 logger.warning(f"Unknown metric: {name}")
        
#         if not metrics:
#             raise ValueError("No valid metrics specified")
        
#         return metrics
    
#     @staticmethod
#     def evaluate_query_engine(
#         query_engine,
#         questions: List[str],
#         eval_dataset: EvaluationDataset,
#         ground_truths: Optional[List[str]] = None,
#         metric_names: Optional[List[str]] = None,
#         **kwargs
#     ) -> Dict[str, Any]:
#         """
#         QueryEngineを評価
        
#         Args:
#             query_engine: LlamaIndexのQueryEngine
#             questions: 評価用の質問リスト
#             ground_truths: 正解データ（オプション）
#             metric_names: 使用するメトリクス名
#             **kwargs: 評価への追加パラメータ
            
#         Returns:
#             評価結果の辞書
#         """
#         try:
#             metrics = RagasEvaluatorFactory.create_metrics(metric_names)

#             logger.info(f"Evaluating with {len(questions)} questions")
#             logger.info(f"Using metrics: {[m.__class__.__name__ for m in metrics]}")
            
#             # Ragas の evaluate 関数を実行
#             result = evaluate(
#                 query_engine=query_engine,
#                 dataset=eval_dataset,
#                 metrics=metrics,
#                 questions=questions,
#                 ground_truths=ground_truths,
#                 llm=Settings.llm,
#                 embeddings=Settings.embed_model,
#                 **kwargs
#             )
            
#             logger.info("Evaluation completed successfully")
#             return result
            
#         except Exception as e:
#             logger.error(f"Evaluation failed: {e}", exc_info=True)
#             raise
    
#     @staticmethod
#     def evaluate_with_local_testset(
#         query_engine,
#         testset_path: str,
#         testset_format: str = "json",
#         metric_names: Optional[List[str]] = None,
#         **kwargs
#     ) -> Dict[str, Any]:
#         """
#         ローカルのテストセットを使って評価
        
#         Args:
#             query_engine: LlamaIndexのQueryEngine
#             testset_path: テストセットファイルパス
#             testset_format: ファイル形式 ("json" or "csv" or "jsonl")
#             metric_names: 使用するメトリクス名
#             **kwargs: 評価への追加パラメータ
            
#         Returns:
#             評価結果の辞書
#         """
#         try:
#             # ローカルテストセットを読み込み
#             testset = RagasEvaluatorFactory.load_local_testset(
#                 dataset_path=testset_path,
#                 format=testset_format
#             )
            
#             # 質問と正解を抽出
#             questions = [item["question"] for item in testset]
#             ground_truths = [item.get("ground_truth") for item in testset]
            
#             # ground_truthsがすべてNoneの場合はNoneを渡す
#             if all(gt is None for gt in ground_truths):
#                 ground_truths = None
            
#             return RagasEvaluatorFactory.evaluate_query_engine(
#                 query_engine=query_engine,
#                 questions=questions,
#                 ground_truths=ground_truths,
#                 metric_names=metric_names,
#                 **kwargs
#             )
            
#         except Exception as e:
#             logger.error(f"Evaluation with local testset failed: {e}", exc_info=True)
#             raise
    
#     @staticmethod
#     def evaluate_with_testset(
#         query_engine,
#         testset: List[Dict[str, Any]],
#         metric_names: Optional[List[str]] = None,
#         **kwargs
#     ) -> Dict[str, Any]:
#         """
#         テストセット（リスト形式）を使って評価
        
#         Args:
#             query_engine: LlamaIndexのQueryEngine
#             testset: テストセットのリスト
#             metric_names: 使用するメトリクス名
#             **kwargs: 評価への追加パラメータ
            
#         Returns:
#             評価結果の辞書
#         """
#         try:
#             # テストセットから質問と正解を抽出
#             questions = [item["question"] for item in testset]
#             ground_truths = [item.get("ground_truth") for item in testset]
            
#             # ground_truthsがすべてNoneの場合はNoneを渡す
#             if all(gt is None for gt in ground_truths):
#                 ground_truths = None
            
#             return RagasEvaluatorFactory.evaluate_query_engine(
#                 query_engine=query_engine,
#                 questions=questions,
#                 ground_truths=ground_truths,
#                 metric_names=metric_names,
#                 **kwargs
#             )
            
#         except Exception as e:
#             logger.error(f"Evaluation with testset failed: {e}", exc_info=True)
#             raise
    
#     @staticmethod
#     def get_evaluation_summary(result: Dict[str, Any]) -> Dict[str, float]:
#         """
#         評価結果のサマリーを取得
        
#         Args:
#             result: evaluate関数の結果
            
#         Returns:
#             メトリクス名と平均スコアの辞書
#         """
#         summary = {}
        
#         # Ragasの結果構造に応じて調整が必要
#         if hasattr(result, "to_pandas"):
#             df = result.to_pandas()
#             for col in df.columns:
#                 if col not in ["question", "contexts", "answer", "ground_truth"]:
#                     summary[col] = float(df[col].mean())
        
#         return summary
    
#     @staticmethod
#     def save_evaluation_results(
#         result: Dict[str, Any],
#         output_path: str,
#         format: str = "json"
#     ):
#         """
#         評価結果を保存
        
#         Args:
#             result: 評価結果
#             output_path: 保存先パス
#             format: 保存形式 ("json" or "csv")
#         """
#         import json
#         from pathlib import Path
        
#         try:
#             output_path = Path(output_path)
#             output_path.parent.mkdir(parents=True, exist_ok=True)
            
#             if format == "json":
#                 # JSON形式で保存
#                 with open(output_path, "w", encoding="utf-8") as f:
#                     json.dump(result, f, ensure_ascii=False, indent=2)
                    
#             elif format == "csv":
#                 # CSV形式で保存
#                 if hasattr(result, "to_pandas"):
#                     df = result.to_pandas()
#                     df.to_csv(output_path, index=False, encoding="utf-8-sig")
#                 else:
#                     raise ValueError("Result cannot be converted to CSV")
            
#             logger.info(f"Evaluation results saved to {output_path}")
            
#         except Exception as e:
#             logger.error(f"Failed to save results: {e}", exc_info=True)
#             raise
