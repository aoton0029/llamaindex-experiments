import logging
import json
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
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
    DatasetGenerator,
    QueryResponseDataset,
)
from llama_index.core.llama_dataset import (
    LabelledRagDataset,
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from .template_prompts import *
from datasets import load_dataset
from llama_index.core.schema import Document, BaseNode
from llama_index.core.llama_pack import download_llama_pack



logger = logging.getLogger(__name__)


class DatasetFactoryBase(ABC):
    def __init__(self):
        self.llm = None
        self.embed_model = None
    
    @abstractmethod
    def generate_dataset_from_documents(self, documents:List[Document], **kwargs):
        pass
    
    @abstractmethod
    def save_dataset_to_json(self, dataset, file_path: str):
        pass

    @abstractmethod
    def load_dataset_from_json(self, file_path: str):
        pass


class EvaluationFactoryBase(ABC):
    def __init__(self):
        self.llm = None
        self.embed_model = None
    
    @abstractmethod
    def create_evaluator(self, **kwargs) -> BaseEvaluator:
        pass

    

class LlamaIndexDatasetFactory(DatasetFactoryBase):
    def __init__(self):
        self.llm = Settings.llm
        self.embed_model = Settings.embed_model
    
    def generate_dataset_from_documents(
        self,
        documents: List[Document],
        num_questions_per_chunk: int = 3,
    ) -> QueryResponseDataset:
        generator = DatasetGenerator.from_documents(
            documents=documents,  
            num_questions_per_chunk=num_questions_per_chunk,
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
                text_question_template=DEFAULT_QUESTION_GENERATION_PROMPT,
                question_gen_query=QUESTION_GEN_QUERY,
                show_progress=True,
            )
        dataset = generator.generate_dataset_from_nodes()
        return dataset
    
    def save_dataset_to_json(self, dataset: QueryResponseDataset, file_path: str):
        dataset.save_json(file_path)

    def load_dataset_from_json(self, file_path: str) -> QueryResponseDataset:
        return QueryResponseDataset.from_json(file_path)


class LlamaIndexEvaluatorFactory(EvaluationFactoryBase):
    def __init__(self):
        self.llm = Settings.llm
        self.embed_model = Settings.embed_model
        self.evaluators = {
            "faithfulness": self._create_faithfulness_evaluator,
            "relevancy": self._create_relevancy_evaluator,
            "correctness": self._create_correctness_evaluator,
            "retriever": self._create_retriever_evaluator,
            "semantic_similarity": self._create_semantic_similarity_evaluator,
            "context_relevancy": self._create_context_relevancy_evaluator,
            "query_response": self._create_query_response_evaluator,
        }


    def create_evaluator(
        self,
        evaluator_type: str,
        **kwargs
    ) -> BaseEvaluator:
        if evaluator_type in self.evaluators:
            return self.evaluators[evaluator_type](**kwargs)
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")

    def _create_faithfulness_evaluator(self, ) -> FaithfulnessEvaluator:
        try:
            return FaithfulnessEvaluator(
                llm=self.llm,
                eval_template=DEFAULT_EVAL_TEMPLATE,
                refine_template=DEFAULT_REFINE_PROMPT,
            )
        except Exception as e:
            logger.error(f"Error creating FaithfulnessEvaluator: {e}")
            raise
    
    def _create_relevancy_evaluator(self) -> RelevancyEvaluator:
        try:
            return RelevancyEvaluator(
                llm=self.llm,
                eval_template=DEFAULT_EVAL_TEMPLATE,
                refine_template=DEFAULT_REFINE_PROMPT,
            )
        except Exception as e:
            logger.error(f"Error creating RelevancyEvaluator: {e}")
            raise
    
    def _create_correctness_evaluator(self) -> CorrectnessEvaluator:
        try:
            return CorrectnessEvaluator(
                llm=self.llm,
                eval_template=DEFAULT_EVAL_TEMPLATE,
                refine_template=DEFAULT_REFINE_PROMPT,
            )
        except Exception as e:
            logger.error(f"Error creating CorrectnessEvaluator: {e}")
            raise
    
    def _create_retriever_evaluator(self) -> RetrieverEvaluator:
        try:
            return RetrieverEvaluator(
                embed_model=self.embed_model,
            )
        except Exception as e:
            logger.error(f"Error creating RetrieverEvaluator: {e}")
            raise
    
    def _create_semantic_similarity_evaluator(self) -> SemanticSimilarityEvaluator:
        try:
            return SemanticSimilarityEvaluator(
                embed_model=self.embed_model,
            )
        except Exception as e:
            logger.error(f"Error creating SemanticSimilarityEvaluator: {e}")
            raise
    
    def _create_context_relevancy_evaluator(self) -> ContextRelevancyEvaluator:
        try:
            return ContextRelevancyEvaluator(
                llm=self.llm,
                eval_template=DEFAULT_EVAL_TEMPLATE,
                refine_template=DEFAULT_REFINE_PROMPT,
            )
        except Exception as e:
            logger.error(f"Error creating ContextRelevancyEvaluator: {e}")
            raise
    
    def _create_query_response_evaluator(self) -> QueryResponseEvaluator:
        try:
            return QueryResponseEvaluator(
                llm=self.llm,
                eval_template=DEFAULT_EVAL_TEMPLATE,
                refine_template=DEFAULT_REFINE_PROMPT,
            )
        except Exception as e:
            logger.error(f"Error creating QueryResponseEvaluator: {e}")
            raise

class LlamaIndexRagDatasetFactory(DatasetFactoryBase):
    def __init__(self):
        super().__init__()
        self.llm = Settings.llm
        self.embed_model = Settings.embed_model

    def generate_dataset_from_documents(
        self,
        documents: List[Document],
        num_questions_per_chunk: int = 3,
    ) -> LabelledRagDataset:
        generator = RagDatasetGenerator.from_documents(
            documents=documents,
            llm=self.llm,
            text_question_template=DEFAULT_QUESTION_GENERATION_PROMPT,
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
            question_gen_query=QUESTION_GEN_QUERY,
            num_questions_per_chunk=num_questions_per_chunk, 
            show_progress=True,
        )
        dataset = generator.generate_dataset_from_nodes()
        return dataset
    
    def save_dataset_to_json(self, dataset: LabelledRagDataset, file_path: str):
        df: pd.DataFrame = dataset.to_pandas()
        df.to_json(file_path, orient="records", lines=True)
    
    def load_dataset_from_json(self, file_path: str) -> LabelledRagDataset:
        dataset = LabelledRagDataset.from_json(file_path)
        return dataset
