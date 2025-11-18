import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from llama_index.core import Settings
from llama_index.core.evaluation import (
    BaseEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner,
    RetrieverEvaluator,
    SemanticSimilarityEvaluator,
    ContextRelevancyEvaluator,
    QueryResponseEvaluator,
    EmbeddingQAFinetuneDataset,
)
from .template_prompts import *
from datasets import load_dataset
from llama_index.core.schema import Document, BaseNode
from llama_index.core.llama_dataset.generator import RagDatasetGenerator, LabelledRagDataset, LabelledRagDataExample
from rag_evaluator.base import RagEvaluatorPack

logger = logging.getLogger(__name__)


class BaseDatasetFactory(ABC):
    """データセット生成のベースクラス"""
    
    @abstractmethod
    def create_dataset_from_document(
        self,
        document: List[Document],
        num_questions: int = 10,
        **kwargs
    ) -> Any:
        """ドキュメントからデータセットを生成"""
        pass

    @abstractmethod
    def create_dataset_from_documents(
        self,
        documents: List[List[Document]],
        num_questions: int = 10,
        **kwargs
    ) -> Any:
        """ディレクトリからドキュメントを読み込みデータセットを生成"""
        pass
    
    @abstractmethod
    def load_dataset_from_json(self, file_path: str) -> Any:
        """ファイルからデータセットを読み込み"""
        pass
    
    @abstractmethod
    def save_dataset_to_json(self, dataset: Any, file_path: str) -> None:
        """データセットをファイルに保存"""
        pass


class BaseEvaluatorFactory(ABC):
    """評価器のベースクラス"""
    
    @abstractmethod
    def create_evaluator(
        self,
        evaluator_type: str,
        **kwargs
    ) -> Any:
        """評価器を作成"""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        dataset: Any,
        query_engine: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """評価を実行"""
        pass


class LlamaIndexDatasetFactory(BaseDatasetFactory):
    """LlamaIndexを使用したデータセット生成ファクトリー"""
    
    def __init__(self):
        self.llm = Settings.llm
        self.embed_model = Settings.embed_model
        logger.info("LlamaIndexDatasetFactory initialized with Settings.llm and Settings.embed_model")
    
    def create_dataset_from_document(
        self,
        documents: List[Document],
        num_questions_per_chunk: int = 3,
        **kwargs
    ) -> LabelledRagDataset:
        logger.info(f"Generating dataset from {len(documents)} documents with {num_questions_per_chunk} questions")
        
        generator = RagDatasetGenerator.from_documents(
            documents=documents,
            llm=self.llm,
            text_question_template=PromptTemplate(JP_QUESTION_GENERATION_PROMPT),
            text_qa_template=JP_TEXT_QA_PROMPT,
            num_questions_per_chunk=num_questions_per_chunk,
            question_gen_query=JP_QUESTION_GEN_QUERY,
            show_progress=True,
        )
        dataset = generator.generate_dataset_from_nodes()
        logger.info(f"Dataset generated with {len(dataset.examples)} examples")
        return dataset
    
        
    def create_dataset_from_documents(
        self,
        documents: List[List[Document]] | List[Document],
        num_questions: int = 10,
        **kwargs
    ) -> LabelledRagDataset:
        logger.info(f"Loading documents from directory {len(documents)} document groups...")

        # フラット化
        all_documents = []
        for docs in documents:
            if isinstance(docs, Document):
                all_documents.append(docs)
            else:
                all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} documents from directory")
        
        # データセット生成
        return self.create_dataset_from_document(
            documents=all_documents,
            num_questions=num_questions,
            **kwargs
        )
        
        
    def load_dataset_from_json(self, file_path: str) -> LabelledRagDataset:
        logger.info(f"Loading dataset from {file_path}")
        fp = Path(file_path)
        if fp.suffix == ".json":
            dataset = LabelledRagDataset.from_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {fp.suffix}")
        logger.info(f"Loaded dataset with {len(dataset.examples)} examples")
        return dataset
    
    
    def save_dataset_to_json(self, dataset: LabelledRagDataset, file_path: str) -> None:
        logger.info(f"Saving dataset to {file_path}")
        fp = Path(file_path)
        if fp.suffix == ".json":
            dataset.save_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {fp.suffix}")
        logger.info(f"Dataset saved with {len(dataset.examples)} examples")

    


class LlamaIndexEvaluatorFactory(BaseEvaluatorFactory):
    """LlamaIndexを使用した評価器ファクトリー"""
    
    def __init__(self):
        self.llm = Settings.llm
        self.embed_model = Settings.embed_model
        logger.info("LlamaIndexEvaluatorFactory initialized with Settings.llm and Settings.embed_model")
    
    def create_evaluator(
        self,
        evaluator_type: str,
        **kwargs
    ) -> BaseEvaluator:
        """
        評価器を作成
        
        Args:
            evaluator_type: 評価器のタイプ
                - "faithfulness": 忠実性
                - "relevancy": 関連性
                - "correctness": 正確性
                - "semantic_similarity": 意味的類似性
                - "context_relevancy": コンテキスト関連性
                - "retriever": リトリーバー評価
            **kwargs: 追加のパラメータ
        
        Returns:
            BaseEvaluator
        """
        logger.info(f"Creating {evaluator_type} evaluator")
        
        evaluator_map = {
            "faithfulness": self._create_faithfulness_evaluator,
            "relevancy": self._create_relevancy_evaluator,
            "correctness": self._create_correctness_evaluator,
            "semantic_similarity": self._create_semantic_similarity_evaluator,
            "context_relevancy": self._create_context_relevancy_evaluator,
        }
        
        if evaluator_type not in evaluator_map:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")

        return evaluator_map[evaluator_type]()

    def _create_faithfulness_evaluator(self) -> FaithfulnessEvaluator:
        return FaithfulnessEvaluator(
            llm=self.llm,
            eval_template=JP_EVAL_TEMPLATE,
            refine_template=JP_REFINE_TEMPLATE
            )

    def _create_relevancy_evaluator(self) -> RelevancyEvaluator:
        return RelevancyEvaluator(
            llm=self.llm,
            eval_template=JP_EVAL_TEMPLATE,
            refine_template=JP_REFINE_TEMPLATE
        )

    def _create_correctness_evaluator(self, score_threshold: float) -> CorrectnessEvaluator:
        return CorrectnessEvaluator(
            llm=self.llm,
            eval_template=JP_EVAL_TEMPLATE,
            score_threshold=score_threshold
        )

    def _create_semantic_similarity_evaluator(self, similarity_mode: str, similarity_threshold: float) -> SemanticSimilarityEvaluator:
        return SemanticSimilarityEvaluator(
            embed_model=self.embed_model,
            similarity_mode=similarity_mode,
            similarity_threshold=similarity_threshold
        )

    def _create_context_relevancy_evaluator(self, score_threshold: float) -> ContextRelevancyEvaluator:
        return ContextRelevancyEvaluator(
            llm=self.llm,
            eval_template=JP_EVAL_TEMPLATE,
            refine_template=JP_REFINE_TEMPLATE,
            score_threshold=score_threshold
        )

    def evaluate(
        self,
        dataset: LabelledRagDataset,
        query_engine: Any = None,
        evaluator_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        LlamaIndexデータセットを使った評価を実行
        # RAG評価例
        rag_evaluator = RagEvaluatorPack(
            query_engine=query_engine,  # built with the same source Documents as the rag_dataset
            rag_dataset=rag_dataset,
        )
        benchmark_df = await rag_evaluator.run()
        
        Args:
            dataset: 評価用データセット
            query_engine: クエリエンジン
            evaluator_types: 使用する評価器のリスト（現在は未使用、RagEvaluatorPackが全評価を実行）
            **kwargs: 追加パラメータ
        
        Returns:
            評価結果の辞書
        """
        logger.info("Starting LlamaIndex evaluation with RagEvaluatorPack")
        
        # 評価を実行
        eval_pack = RagEvaluatorPack(
            rag_dataset=dataset,
            query_engine=query_engine,
            judge_llm=self.llm,
            embed_model=self.embed_model,
            show_progress=kwargs.get("show_progress", True)
        )

        df_result = eval_pack.run()
        logger.info(f"LlamaIndex evaluation completed: {len(df_result)} results")
        
        return df_result.to_dict("records")
