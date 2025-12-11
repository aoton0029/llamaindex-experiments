from typing import Optional, Dict, Any
from llama_index.core import Settings
from llama_index.core.llms.llm import BaseLLM
from ..factories.llm_factory import LLMFactory
import logging

logger = logging.getLogger(__name__)


class _DomainLLMSettings:
    """ドメイン別LLM設定"""
    
    _llm_config: Dict[str, Any] = {}
    _domain_kwargs: Dict[str, Dict[str, Any]] = {}
    _domain_llm_cache: Dict[str, BaseLLM] = {}
    _loaded: bool = False
    
    def initialize(self, llm_config: Dict[str, Any], domain_kwargs: Dict[str, Any]):
        """ドメイン設定を初期化
        
        Args:
            llm_config: LLMモデルの基本設定（backend, model_name, base_url等）
            domain_kwargs: ドメイン別のLLMパラメータ設定
        """
        self._loaded = False
        self._llm_config = llm_config
        self._domain_kwargs = domain_kwargs
        self._domain_llm_cache.clear()
        self._loaded = True
        self._create_domain_llm_instances()    

    def _create_domain_llm_instances(self):
        """全ドメインのLLMインスタンスを事前生成してキャッシュ"""
        if not self._loaded or not self._llm_config:
            return
        
        # llm_configからLLMの基本設定を取得
        backend = self._llm_config.get("backend")
        model_name = self._llm_config.get("model_name")
        base_url = self._llm_config.get("base_url")
        
        if not all([backend, model_name, base_url]):
            logger.error("Invalid llm_config: missing backend, model_name, or base_url")
            return
        
        for domain_name, kwargs in self._domain_kwargs.items():
            try:
                # additional_kwargsを処理
                additional_kwargs = kwargs.get("additional_kwargs", {})
                
                # LLMインスタンスを新規作成（パラメータをオーバーライド）
                llm_kwargs = {
                    "temperature": kwargs.get("temperature", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "timeout": kwargs.get("timeout", 180),
                    "additional_kwargs": additional_kwargs,
                }
                
                # LLMFactoryを使って新しいインスタンスを作成
                llm_instance = LLMFactory.create(
                    backend=backend,
                    model_name=model_name,
                    base_url=base_url,
                    **llm_kwargs
                )
                self._domain_llm_cache[domain_name] = llm_instance
                logger.debug(f"Created LLM instance for domain '{domain_name}'")
            except Exception as e:
                logger.warning(f"Failed to create LLM instance for domain '{domain_name}': {e}")
        
        logger.info(f"Created {len(self._domain_llm_cache)} domain LLM instances")
    
    def get_domain_info(self) -> dict:
        return {
            "loaded": self._loaded,
            "available_domains": list(self._domain_kwargs.keys()),
            "domain_configs": self._domain_kwargs,
        }

    def _get_domain_llm(self, domain_name: str) -> BaseLLM:
        """指定ドメインのキャッシュ済みLLMを取得"""
        # キャッシュから取得
        if domain_name in self._domain_llm_cache:
            return self._domain_llm_cache[domain_name]
        
        # デフォルトにフォールバック
        if "default" in self._domain_llm_cache:
            logger.warning(f"No cached LLM for domain '{domain_name}', using default")
            return self._domain_llm_cache["default"]
        
        # 最終フォールバック
        logger.warning(f"No configuration found for domain '{domain_name}', using Settings.llm")
        return Settings.llm
    
    # ==========================================
    # デフォルト設定
    # ==========================================
    @property
    def DEFAULT(self) -> BaseLLM:
        """デフォルト設定のLLMを取得"""
        return self._get_domain_llm("default")
    
    # ==========================================
    # Extractor系
    # ==========================================
    @property
    def EXTRACTOR_TITLE(self) -> BaseLLM:
        """TitleExtractor用のLLM"""
        return self._get_domain_llm("extractor_title")
    
    @property
    def EXTRACTOR_SUMMARY(self) -> BaseLLM:
        """SummaryExtractor用のLLM"""
        return self._get_domain_llm("extractor_summary")
    
    @property
    def EXTRACTOR_KEYWORD(self) -> BaseLLM:
        """KeywordExtractor用のLLM"""
        return self._get_domain_llm("extractor_keyword")
    
    @property
    def EXTRACTOR_QA(self) -> BaseLLM:
        """QuestionsAnsweredExtractor用のLLM"""
        return self._get_domain_llm("extractor_qa")
    
    # ==========================================
    # Selector系
    # ==========================================
    @property
    def SELECTOR(self) -> BaseLLM:
        """LLMSingleSelector/LLMMultiSelector用のLLM"""
        return self._get_domain_llm("selector")
    
    # ==========================================
    # Response Synthesizer系
    # ==========================================
    @property
    def SYNTHESIZER_RESPONSE(self) -> BaseLLM:
        """ResponseSynthesizer用のLLM"""
        return self._get_domain_llm("synthesizer_response")
    
    @property
    def SYNTHESIZER_TREE_SUMMARIZE(self) -> BaseLLM:
        """TreeSummarize用のLLM"""
        return self._get_domain_llm("synthesizer_tree_summarize")
    
    @property
    def SYNTHESIZER_REFINE(self) -> BaseLLM:
        """Refine用のLLM"""
        return self._get_domain_llm("synthesizer_refine")
    
    # ==========================================
    # Index別の設定
    # ==========================================
    def get_by_index(self, index_type: str) -> BaseLLM:
        """指定インデックスタイプ用のLLMを取得"""
        domain_map = {
            "VectorStoreIndex": "index_vector_store",
            "SummaryIndex": "index_summary",
            "TreeIndex": "index_tree",
            "KeywordTableIndex": "index_keyword_table",
            "KnowledgeGraphIndex": "index_knowledge_graph",
            "DocumentSummaryIndex": "index_document_summary",
        }
        domain_name = domain_map.get(index_type)
        if domain_name:
            return self._get_domain_llm(domain_name)
        else:
            logger.warning(f"Unknown index type '{index_type}', using default LLM")
            return self.DEFAULT

    @property
    def INDEX_VECTOR_STORE(self) -> BaseLLM:
        """VectorStoreIndex用のLLM"""
        return self._get_domain_llm("index_vector_store")
    
    @property
    def INDEX_SUMMARY(self) -> BaseLLM:
        """SummaryIndex用のLLM"""
        return self._get_domain_llm("index_summary")
    
    @property
    def INDEX_TREE(self) -> BaseLLM:
        """TreeIndex用のLLM"""
        return self._get_domain_llm("index_tree")
    
    @property
    def INDEX_KEYWORD_TABLE(self) -> BaseLLM:
        """KeywordTableIndex用のLLM"""
        return self._get_domain_llm("index_keyword_table")
    
    @property
    def INDEX_KNOWLEDGE_GRAPH(self) -> BaseLLM:
        """KnowledgeGraphIndex用のLLM"""
        return self._get_domain_llm("index_knowledge_graph")
    
    @property
    def INDEX_DOCUMENT_SUMMARY(self) -> BaseLLM:
        """DocumentSummaryIndex用のLLM"""
        return self._get_domain_llm("index_document_summary")

    # ==========================================
    # Query Engine別の設定
    # ==========================================
    @property
    def QUERY_ENGINE_ROUTER_SELECTOR(self) -> BaseLLM:
        """RouterQueryEngine用のLLM"""
        return self._get_domain_llm("query_engine_router_selector")
    
    @property
    def QUERY_ENGINE_SUB_QUESTION(self) -> BaseLLM:
        """SubQuestionQueryEngine用のLLM"""
        return self._get_domain_llm("query_engine_sub_question")
    
    @property
    def QUERY_ENGINE_RETRIEVER(self) -> BaseLLM:
        """RetrieverQueryEngine用のLLM"""
        return self._get_domain_llm("query_engine_retriever")
    
    @property
    def QUERY_ENGINE_MULTI_STEP(self) -> BaseLLM:
        """MultiStepQueryEngine用のLLM"""
        return self._get_domain_llm("query_engine_multi_step")
    


DomainLLMSettings = _DomainLLMSettings() 