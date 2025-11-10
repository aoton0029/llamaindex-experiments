import logging
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
from .template_prompts import *
from datasets import load_dataset
from llama_index.core.schema import Document, BaseNode

logger = logging.getLogger(__name__)


class DatasetFactory:
    @staticmethod
    def create_dataset_generator(
        documents: List[Document],
        num_questions_per_chunk: int = 10
    ) -> DatasetGenerator:
        """
        DatasetGeneratorを作成
        
        Args:
            documents: 質問生成に使用するドキュメントのリスト
            num_questions_per_chunk: チャンクあたりの質問数
        """
        generator = DatasetGenerator.from_documents(
            documents,  
            num_questions_per_chunk=num_questions_per_chunk,
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
            text_question_template=DEFAULT_QUESTION_GENERATION_PROMPT,
            question_gen_query=QUESTION_GEN_QUERY,
        )
        return generator
    
    @staticmethod
    def auto_generate_dataset(
        documents: List[Document],
        num_questions: int = 50,
        num_questions_per_chunk: int = 2,
        save_path: Optional[str] = None,
        include_contexts: bool = True,
        question_types: Optional[List[str]] = None,
    ) -> QueryResponseDataset:
        """
        ドキュメントから自動的にデータセットを生成
        
        Args:
            documents: ドキュメントのリスト
            num_questions: 生成する質問の総数
            num_questions_per_chunk: チャンクあたりの質問数
            save_path: データセットを保存するパス（オプション）
            include_contexts: コンテキスト情報を含めるか
            question_types: 生成する質問のタイプ（例: ['factual', 'analytical', 'comparative']）
        """
        logger.info(f"自動データセット生成開始: {num_questions}問を生成")
        
        try:
            # DatasetGeneratorを作成
            generator = DatasetFactory.create_dataset_generator(
                documents=documents,
                num_questions_per_chunk=num_questions_per_chunk
            )
            
            # データセット生成
            dataset = generator.generate_dataset_from_nodes(num=num_questions)
            
            # 質問タイプのフィルタリング（オプション）
            if question_types:
                dataset = DatasetFactory._filter_by_question_types(
                    dataset, question_types
                )
            
            # コンテキスト情報の追加
            if include_contexts:
                dataset = DatasetFactory._add_contexts_to_dataset(
                    dataset, documents
                )
            
            # データセットの保存
            if save_path:
                DatasetFactory.save_dataset(dataset, save_path)
                logger.info(f"データセットを保存しました: {save_path}")
            
            logger.info(f"データセット生成完了: {len(dataset.queries)}問")
            return dataset
            
        except Exception as e:
            logger.error(f"データセット生成中にエラーが発生: {str(e)}")
            raise
    
    @staticmethod
    def auto_generate_from_index(
        index,
        num_questions: int = 50,
        save_path: Optional[str] = None,
        include_contexts: bool = True,
    ) -> QueryResponseDataset:
        """
        インデックスから自動的にデータセットを生成
        
        Args:
            index: LlamaIndexのインデックス
            num_questions: 生成する質問数
            save_path: 保存先パス
            include_contexts: コンテキストを含めるか
        """
        logger.info("インデックスからデータセット生成開始")
        
        try:
            # インデックスからドキュメントを取得
            docstore = index.docstore
            documents = [docstore.get_document(doc_id) for doc_id in docstore.get_all_document_hashes()]
            
            # データセット生成
            dataset = DatasetFactory.auto_generate_dataset(
                documents=documents,
                num_questions=num_questions,
                save_path=save_path,
                include_contexts=include_contexts,
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"インデックスからのデータセット生成エラー: {str(e)}")
            raise
    
    @staticmethod
    def generate_multilingual_dataset(
        documents: List[Document],
        languages: List[str] = ["ja", "en"],
        num_questions_per_language: int = 25,
        save_path: Optional[str] = None,
    ) -> Dict[str, QueryResponseDataset]:
        """
        多言語データセットを生成
        
        Args:
            documents: ドキュメントのリスト
            languages: 対象言語のリスト
            num_questions_per_language: 言語ごとの質問数
            save_path: 保存先パス
        """
        logger.info(f"多言語データセット生成開始: {languages}")
        
        datasets = {}
        for lang in languages:
            logger.info(f"{lang}のデータセット生成中...")
            
            # 言語別のプロンプトを設定（必要に応じて）
            dataset = DatasetFactory.auto_generate_dataset(
                documents=documents,
                num_questions=num_questions_per_language,
            )
            
            datasets[lang] = dataset
            
            if save_path:
                lang_save_path = str(Path(save_path).parent / f"{Path(save_path).stem}_{lang}.json")
                DatasetFactory.save_dataset(dataset, lang_save_path)
        
        logger.info("多言語データセット生成完了")
        return datasets
    
    @staticmethod
    def _filter_by_question_types(
        dataset: QueryResponseDataset,
        question_types: List[str]
    ) -> QueryResponseDataset:
        """質問タイプでデータセットをフィルタリング"""
        # 実装は質問タイプの分類ロジックに依存
        # ここでは簡易的な実装
        return dataset
    
    @staticmethod
    def _add_contexts_to_dataset(
        dataset: QueryResponseDataset,
        documents: List[Document]
    ) -> QueryResponseDataset:
        """データセットにコンテキスト情報を追加"""
        if not dataset.reference_contexts:
            reference_contexts = {}
            for query_id in dataset.queries.keys():
                # 各質問に対応するコンテキストを検索
                # 簡易実装：最初のドキュメントのテキストを使用
                if documents:
                    reference_contexts[query_id] = [documents[0].text]
            
            dataset.reference_contexts = reference_contexts
        
        return dataset
    
    @staticmethod
    def save_dataset(
        dataset: QueryResponseDataset,
        save_path: str,
        format: str = "json"
    ) -> None:
        """
        データセットをファイルに保存
        
        Args:
            dataset: 保存するデータセット
            save_path: 保存先パス
            format: 保存形式（'json', 'jsonl'）
        """
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "queries": dataset.queries,
            "responses": dataset.responses,
        }
        
        if dataset.reference_contexts:
            data["reference_contexts"] = dataset.reference_contexts
        
        if format == "json":
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(save_path, 'w', encoding='utf-8') as f:
                for query_id in dataset.queries.keys():
                    item = {
                        "query_id": query_id,
                        "query": dataset.queries[query_id],
                        "response": dataset.responses[query_id],
                    }
                    if dataset.reference_contexts:
                        item["contexts"] = dataset.reference_contexts.get(query_id, [])
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"データセットを保存: {save_path}")
    
    @staticmethod
    def load_dataset_from_file(
        file_path: str,
        format: str = "json"
    ) -> QueryResponseDataset:
        """
        ファイルからデータセットを読み込み
        
        Args:
            file_path: ファイルパス
            format: ファイル形式（'json', 'jsonl'）
        """
        if format == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif format == "jsonl":
            data = {"queries": {}, "responses": {}, "reference_contexts": {}}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    query_id = item["query_id"]
                    data["queries"][query_id] = item["query"]
                    data["responses"][query_id] = item["response"]
                    if "contexts" in item:
                        data["reference_contexts"][query_id] = item["contexts"]
        
        dataset = QueryResponseDataset(
            queries=data["queries"],
            responses=data["responses"],
            reference_contexts=data.get("reference_contexts"),
        )
        
        logger.info(f"データセットを読み込み: {file_path}")
        return dataset
        
    @staticmethod
    def generate_dataset_from_nodes(
        nodes: List[BaseNode],
        num_questions: int = 10
    ) -> QueryResponseDataset:
        """
        ノードから質問と回答のデータセットを生成
        
        Args:
            nodes: ノードのリスト
            num_questions: 生成する質問の数
        """
        generator = DatasetGenerator(
            nodes=nodes,
            num_questions_per_chunk=2,
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
            text_question_template=DEFAULT_QUESTION_GENERATION_PROMPT,
            question_gen_query=QUESTION_GEN_QUERY,
        )
        dataset = generator.generate_dataset_from_nodes(num=num_questions)
        return dataset
    
    @staticmethod
    def generate_query_response_dataset(
        documents: List[Document],
        num_questions: int = 10
    ) -> QueryResponseDataset:
        """
        ドキュメントから質問と回答のデータセットを生成
        
        Args:
            documents: ドキュメントのリスト
            llm: 使用するLLM
            num_questions: 生成する質問の数
        """
        generator = DatasetFactory.create_dataset_generator(
            documents=documents,
        )
        dataset = generator.generate_dataset_from_nodes(num=num_questions)
        return dataset

    
    @staticmethod
    def evaluate_dataset(
        dataset: QueryResponseDataset,
        evaluator_types: List[str],
    ) -> Dict[str, Any]:
        evaluators = []
        for evaluator_type in evaluator_types:
            evaluator = EvaluatorFactory.create_evaluator(evaluator_type)
            if evaluator:
                evaluators.append(evaluator)
        
        if not evaluators:
            logger.error("No valid evaluators provided for dataset evaluation.")
            return {}
        
        batch_runner = BatchEvalRunner(evaluators=evaluators, show_progress=True)
        results = batch_runner.evaluate_responses(dataset=dataset)
        return results

    @staticmethod
    def load_huggingface_dataset(
        dataset_name: str,
        split: str = "train",
        query_field: str = "query",
        answer_field: str = "answer",
        context_field: Optional[str] = None,
    ) -> QueryResponseDataset:
        """
        HuggingFaceのデータセットをLlamaIndexのQueryResponseDatasetに変換
        
        Args:
            dataset_name: HuggingFaceのデータセット名
            split: データセットの分割('train', 'test', 'validation')
            query_field: クエリのフィールド名
            answer_field: 回答のフィールド名
            context_field: コンテキストのフィールド名(オプション)
        """
        # HuggingFaceデータセットをロード
        hf_dataset = load_dataset(dataset_name, split=split)
        
        # QueryResponseDatasetに変換
        queries = {}
        responses = {}
        reference_contexts = {} if context_field else None
        
        for idx, item in enumerate(hf_dataset):
            query_id = f"query_{idx}"
            queries[query_id] = item[query_field]
            responses[query_id] = item[answer_field]
            
            if context_field and context_field in item:
                reference_contexts[query_id] = [item[context_field]]
        
        # QueryResponseDatasetを作成
        dataset = QueryResponseDataset(
            queries=queries,
            responses=responses,
            reference_contexts=reference_contexts,
        )
        
        return dataset




class EvaluatorFactory:
    @staticmethod
    def create_evaluator(evaluator_type: str, **kwargs) -> Optional[BaseEvaluator]:
        if evaluator_type == "faithfulness":
            return EvaluatorFactory.create_faithfulness_evaluator(**kwargs)
        elif evaluator_type == "relevancy":
            return EvaluatorFactory.create_relevancy_evaluator(**kwargs)
        elif evaluator_type == "correctness":
            return EvaluatorFactory.create_correctness_evaluator(**kwargs)
        elif evaluator_type == "semantic_similarity":
            return EvaluatorFactory.create_semantic_similarity_evaluator(**kwargs)
        elif evaluator_type == "context_relevancy":
            return EvaluatorFactory.create_context_relevancy_evaluator(**kwargs)
        else:
            logger.warning(f"未知の評価者タイプ: {evaluator_type}")
            return None
    
    @staticmethod
    def create_faithfulness_evaluator() -> FaithfulnessEvaluator:
        evaluator = FaithfulnessEvaluator(
            eval_template=DEFAULT_EVAL_TEMPLATE,
            refine_template=DEFAULT_REFINE_TEMPLATE,
        )
        return evaluator
    
    @staticmethod
    def create_relevancy_evaluator() -> RelevancyEvaluator:
        evaluator = RelevancyEvaluator(
            eval_template=DEFAULT_EVAL_TEMPLATE,
            refine_template=DEFAULT_REFINE_TEMPLATE,
        )
        return evaluator
    
    @staticmethod
    def create_correctness_evaluator(score_threshold: float) -> CorrectnessEvaluator:
        evaluator = CorrectnessEvaluator(
            eval_template=DEFAULT_EVAL_TEMPLATE,
            score_threshold=score_threshold,
        )
        return evaluator
    
    @staticmethod
    def create_semantic_similarity_evaluator(similarity_threshold: float, similarity_mode: str) -> SemanticSimilarityEvaluator:
        from llama_index.core.base.embeddings.base import SimilarityMode, similarity
        evaluator = SemanticSimilarityEvaluator(            
            similarity_threshold=similarity_threshold,
            similarity_mode=SimilarityMode[similarity_mode],
        )
        return evaluator
    
    @staticmethod
    def create_context_relevancy_evaluator() -> ContextRelevancyEvaluator:
        evaluator = ContextRelevancyEvaluator(
            eval_template=DEFAULT_EVAL_TEMPLATE,
            refine_template=DEFAULT_REFINE_TEMPLATE,
        )
        return evaluator
    
    @staticmethod
    def create_query_response_evaluator() -> QueryResponseEvaluator:
        evaluator = QueryResponseEvaluator(
            eval_template=DEFAULT_EVAL_TEMPLATE,
            refine_template=DEFAULT_REFINE_TEMPLATE,
        )
        return evaluator
    
