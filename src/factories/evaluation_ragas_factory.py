import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core import Settings
from llama_index.core.schema import Document
from datasets import load_dataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    AnswerSimilarity,
    AnswerCorrectness,
)
from ragas.integrations.llama_index import evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.dataset import Dataset
from ragas import EvaluationDataset
from ragas.testset.persona import Persona
from ragas.testset.graph import KnowledgeGraph
from .evaluation_factory import DatasetFactoryBase, EvaluationFactoryBase


logger = logging.getLogger(__name__)


class RagasDatasetFactory(DatasetFactoryBase):
    def __init__(self):
        self.llm = LlamaIndexLLMWrapper(Settings.llm)
        self.embed_model = LlamaIndexEmbeddingsWrapper(Settings.embed_model)

    def generate_dataset_from_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> EvaluationDataset:
        
        return dataset

    def save_dataset_to_json(self, dataset: EvaluationDataset, file_path: str):
        dataset.to_json(file_path)

    def load_dataset_from_json(self, file_path: str) -> EvaluationDataset:
        return EvaluationDataset.from_json(file_path)


class RagasEvaluatorFactory(EvaluationFactoryBase):
    def __init__(self):
        self.llm = LlamaIndexLLMWrapper(Settings.llm)
        self.embed_model = LlamaIndexEmbeddingsWrapper(Settings.embed_model)
        self.available_metrics = {
            "faithfulness": self._create_faithfulness_evaluator,
            "answer_relevancy": self._create_answer_relevancy_evaluator,
            "context_precision": self._create_context_precision_evaluator,
            "context_recall": self._create_context_recall_evaluator,
            "context_relevance": self._create_context_relevancy_evaluator,
            "answer_similarity": self._create_semantic_similarity_evaluator,
            "answer_correctness": self._create_correctness_evaluator,
        }

    def create_evaluators(self, metric_names: Optional[Dict[str, Any]] = None) -> List[Any]:
        try:
            evaluators = []
            if metric_names is None:
                metric_names = list(self.available_metrics.keys())
            for name in metric_names:
                if name in self.available_metrics:
                    evaluator = self.available_metrics[name]()
                    evaluators.append(evaluator)
                else:
                    logger.warning(f"Unknown evaluator metric: {name}")
            return evaluators
        except Exception as e:
            logger.error(f"Error creating evaluators: {e}")
            raise

    def _create_faithfulness_evaluator(self, **kwargs) -> Faithfulness:
        try:
            return Faithfulness(llm=self.llm)
        except Exception as e:
            logger.error(f"Faithfulness evaluatorの作成エラー: {e}")
            raise
    
    def _create_answer_relevancy_evaluator(self, **kwargs) -> AnswerRelevancy:
        try:
            return AnswerRelevancy(llm=self.llm, embeddings=self.embed_model)
        except Exception as e:
            logger.error(f"AnswerRelevancy evaluatorの作成エラー: {e}")
            raise

    def _create_correctness_evaluator(self, **kwargs) -> AnswerCorrectness:
        try:
            return AnswerCorrectness(llm=self.llm)
        except Exception as e:
            logger.error(f"AnswerCorrectness evaluatorの作成エラー: {e}")
            raise

    def _create_semantic_similarity_evaluator(self, **kwargs) -> AnswerSimilarity:
        try:
            return AnswerSimilarity(embeddings=self.embed_model)
        except Exception as e:
            logger.error(f"AnswerSimilarity evaluatorの作成エラー: {e}")
            raise
    
    def _create_context_relevancy_evaluator(self, **kwargs) -> ContextRelevance:
        try:
            return ContextRelevance(llm=self.llm)
        except Exception as e:
            logger.error(f"ContextRelevance evaluatorの作成エラー: {e}")
            raise
    
    def _create_context_recall_evaluator(self, **kwargs) -> ContextRecall:
        try:
            return ContextRecall(llm=self.llm)
        except Exception as e:
            logger.error(f"ContextRecall evaluatorの作成エラー: {e}")
            raise
    
    def _create_context_precision_evaluator(self, **kwargs) -> ContextPrecision:
        try:
            return ContextPrecision(llm=self.llm)
        except Exception as e:
            logger.error(f"ContextPrecision evaluatorの作成エラー: {e}")
            raise
    
    
    
    