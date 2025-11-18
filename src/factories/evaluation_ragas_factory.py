import logging
import json
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import Settings
from llama_index.core.schema import Document
from datasets import load_dataset, Dataset as HFDataset
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    AnswerSimilarity,
    AnswerCorrectness,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper, LlamaIndexLLMWrapper, BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper, BaseRagasEmbeddings
from ragas.testset.persona import Persona
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor, apply_transforms
from ragas.callbacks import RagasTracer
from .evaluation_factory import BaseDatasetFactory, BaseEvaluatorFactory
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)


class RagasDatasetFactory(BaseDatasetFactory):
    """
    Ragasを使用したデータセット生成ファクトリー
    
    日本語評価のための設定例:
    - ペルソナ、NER抽出器、クエリ合成器を日本語対応にする
    """
    
    def __init__(self):
        self.language = "japanese"
        self.llm = (Settings.llm)
        self.embeddings = LlamaIndexEmbeddingsWrapper(Settings.embed_model)
        
        self.persona_list = [
            Persona(
                name="System Manager of General Company",
                role_description="I am a systems manager at a general company and don't know much about AI, but I have received a proposal for a system that uses AI from an AI vendor and I am trying to understand the details.",
            ),
            # Persona(
            #     name="Researcher",
            #     role_description="研究者です。日本語で専門的な質問をします。",
            # ),
        ]

        logger.info(
            f"RagasDatasetFactory initialized with language={self.language}, "
            f"{len(self.persona_list)} personas"
        )
    
    def _create_knowledge_graph(self, docs: List[Document]):
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type = NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.get_content(),
                        "document_metadata": doc.metadata
                    }
                )
            )
        return kg
    
    async def _create_transoforms(self):
        # NER抽出器の設定
        transforms = [
            HeadlinesExtractor(llm=self.llm, max_num=20),
            # HeadlineSplitter(max_tokens=1500),
            KeyphrasesExtractor(llm=self.llm),
            NERExtractor(llm=self.llm)
        ]
        for transform in transforms:
            prompts = await transform.adapt_prompts(
                language=self.language,
                llm=self.llm
            )
            transform.set_prompts(**prompts)
        return transforms
    
    async def _create_distribution(self):
        # クエリ合成器の設定
        distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.llm), 1.0)
        ]
        for query, _ in distribution:
            prompts = await query.adapt_prompts(self.language, llm=self.llm)
            query.set_prompts(**prompts)
        
        return distribution 
        
    
    async def create_dataset_from_document(
        self,
        documents: List[Document],
        num_questions: int = 10,
        **kwargs
    ) -> EvaluationDataset:
        logger.info(
            f"Generating Ragas dataset from {len(documents)} documents "
            f"with {num_questions} questions in {self.language}"
        )
        
        # TestsetGeneratorを作成
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            persona_list=self.persona_list,
        )
        
        transforms = await self._create_transoforms()
        distribution = await self._create_distribution()
        
        # データセットを生成
        testset = generator.generate_with_llamaindex_docs(
            documents=documents,
            testset_size=num_questions,
            transforms=transforms,
            transforms_llm=self.llm,
            transforms_embedding_model=self.embeddings,
            query_distribution=distribution,
            callbacks=RagasTracer(),
        )
        
        logger.info(f"Ragas dataset generated with {len(testset)} samples")
        return testset
    
    def load_dataset_from_json(self, file_path: str) -> EvaluationDataset:
        logger.info(f"Loading Ragas dataset from {file_path}")
        fp = Path(file_path)
        if fp.suffix == ".json":
            dataset = EvaluationDataset.from_pandas(pd.read_json(file_path))
        else:
            raise ValueError(f"Unsupported file type: {fp.suffix}")
        logger.info(f"Loaded Ragas dataset with {len(dataset)} samples")
        return dataset
    
    def save_dataset_to_json(self, dataset: EvaluationDataset, file_path: str) -> None:
        logger.info(f"Saving Ragas dataset to {file_path}")
        fp = Path(file_path)
        if fp.suffix == ".json":
            df = dataset.to_pandas()
            df.to_json(path_or_buf=file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file type: {fp.suffix}")
        logger.info(f"Saved Ragas dataset with {len(dataset)} samples")
    
    async def create_dataset_from_documents(
        self,
        documents: List[List[Document]],
        num_questions: int = 10,
        **kwargs
    ) -> EvaluationDataset:
        logger.info(f"Loading documents from directory {len(documents)} document groups...")

        # フラット化
        all_documents = []
        for docs in documents:
            all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} documents from directory")
        
        # データセット生成
        return await self.create_dataset_from_document(
            documents=all_documents,
            num_questions=num_questions,
            **kwargs
        )


class RagasEvaluatorFactory(BaseEvaluatorFactory):
    """Ragasを使用した評価器ファクトリー"""
    
    def __init__(self):
        self.language = "japanese"
        self.llm = LlamaIndexLLMWrapper(Settings.llm)
        self.embeddings = LlamaIndexEmbeddingsWrapper(Settings.embed_model)
        logger.info(f"RagasEvaluatorFactory initialized with language={self.language}")

    async def _adapt_metric_to_language(self, metric):
        """メトリクスを指定言語に適応"""
        if hasattr(metric, 'adapt'):
            adapted_metric = await metric.adapt(
                language=self.language,
                llm=self.llm
            )
            return adapted_metric
        return metric
    
    def create_evaluator(
        self,
        evaluator_type: str,
        **kwargs
    ) -> Any:
        """
        Ragasの評価メトリクスを作成
        
        Args:
            evaluator_type: 評価器のタイプ
                - "faithfulness": 忠実性
                - "answer_relevancy": 回答関連性
                - "context_precision": コンテキスト精度
                - "context_recall": コンテキスト再現率
                - "context_entity_recall": コンテキストエンティティ再現率
                - "answer_similarity": 回答類似性
                - "answer_correctness": 回答正確性
            **kwargs: 追加のパラメータ
        
        Returns:
            Ragasメトリクス
        """
        logger.info(f"Creating Ragas {evaluator_type} metric")
        
        metric_map = {
            "faithfulness": self._create_faithfulness_metric,
            "answer_relevancy": self._create_answer_relevancy_metric,
            "context_precision": self._create_context_precision_metric,
            "context_recall": self._create_context_recall_metric,
            "context_entity_recall": self._create_context_entity_recall_metric,
            "answer_similarity": self._create_answer_similarity_metric,
            "answer_correctness": self._create_answer_correctness_metric,
        }
        
        if evaluator_type not in metric_map:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")
        
        metric = metric_map[evaluator_type]()
        
        # 日本語に適応
        if self.language != "english":
            metric = asyncio.run(self._adapt_metric_to_language(metric))
        
        return metric
    
    def _create_faithfulness_metric(self, **kwargs) -> Faithfulness:
        return Faithfulness(llm=self.llm)
    
    def _create_answer_relevancy_metric(self, **kwargs) -> AnswerRelevancy:
        return AnswerRelevancy(
            llm=self.llm,
            embeddings=self.embeddings
        )
    
    def _create_context_precision_metric(self, **kwargs) -> ContextPrecision:
        return ContextPrecision(llm=self.llm)
    
    def _create_context_recall_metric(self, **kwargs) -> ContextRecall:
        return ContextRecall(llm=self.llm)
    
    def _create_context_entity_recall_metric(self, **kwargs) -> ContextEntityRecall:
        return ContextEntityRecall(llm=self.llm)
    
    def _create_answer_similarity_metric(self, **kwargs) -> AnswerSimilarity:
        return AnswerSimilarity(embeddings=self.embeddings)
    
    def _create_answer_correctness_metric(self, **kwargs) -> AnswerCorrectness:
        return AnswerCorrectness(llm=self.llm)

    def evaluate(
        self,
        dataset: EvaluationDataset,
        query_engine: Any = None,
        evaluator_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ragasデータセットを使った評価を実行
        
        Args:
            dataset: Ragas評価用データセット
            query_engine: クエリエンジン
            evaluator_types: 使用する評価器のリスト
            **kwargs: 追加パラメータ
        
        Returns:
            評価結果の辞書
        """
        logger.info(f"Starting Ragas evaluation with {evaluator_types}")
        
        if not evaluator_types:
            evaluator_types = ["faithfulness", "answer_relevancy"]
        
        # メトリクスを作成
        metrics = []
        for evaluator_type in evaluator_types:
            metric = self.create_evaluator(evaluator_type, **kwargs)
            metrics.append(metric)

        # クエリエンジンを使って評価データセットを完成させる
        # (query_engineでresponseとcontextsを取得)
        if query_engine:
            logger.info("Generating responses using query engine")
            samples = []
            for sample in dataset:
                response = query_engine.query(sample.user_input)
                
                # SingleTurnSampleを作成
                from ragas import SingleTurnSample
                new_sample = SingleTurnSample(
                    user_input=sample.user_input,
                    response=str(response),
                    retrieved_contexts=[node.get_content() for node in response.source_nodes] if hasattr(response, 'source_nodes') else [],
                    reference=sample.reference if hasattr(sample, 'reference') else None,
                )
                samples.append(new_sample)
            
            # 新しいデータセットを作成
            dataset = EvaluationDataset(samples=samples)
        
        # 評価を実行
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )
        
        # 結果を辞書形式に変換
        df_result = result.to_pandas()

        logger.info(f"Ragas evaluation completed")
        return df_result.to_dict("records")

