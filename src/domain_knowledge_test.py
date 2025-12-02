from pathlib import Path
from typing import Dict, Any, List
from llama_index.core import Settings, Document
from llama_index.core.ingestion import IngestionPipeline
from test_runner_base import TestRunnerBase
from models import (
    GlossaryTerm,
    TechColumnTerm,
    PdfDocumentVector,
)
from factories import (
    DocumentLoader,
)
from factories.template_prompts import TemplatePromptSettings
from factories.retriever_factory import RetrieverFactory
from factories.query_engine_factory import QueryEngineFactory
from factories.post_processor_factory import PostProcessorFactory 


class StoreDomainKnowledgeRunner(TestRunnerBase):
    """ドメイン知識をストレージに保存するテストランナー"""
    
    def _get_milvus_schema(self, context_name: str, dim: int):
        """コンテキスト名に応じたMilvusスキーマを取得"""
        if "glossary" in context_name.lower():
            return GlossaryTerm.get_milvus_schema(dim)
        elif "tech_column" in context_name.lower():
            return TechColumnTerm.get_milvus_schema(dim)
        else:
            raise ValueError(f"Unknown context name for schema: {context_name}")
    
    def run_pattern(self, pattern_name: str):
        """
        指定されたパターンに従ってドメイン知識をストレージに保存
        
        Args:
            pattern_name: テストパターン名
        """
        try: 
            TemplatePromptSettings.initialize(self.config_manager.get_template_prompts())
            
            pattern_dict = self.test_pattern_manager.get_store_domain_test_pattern(pattern_name)

            self.monitor.start_test(pattern_name, pattern_dict)

            # パターン設定の読み込み
            self.monitor.log_event("setup", f"Loading pattern: {pattern_name}")
            
            if not pattern_dict:
                raise ValueError(f"Pattern not found: {pattern_name}")
            
            if not pattern_dict.get("enabled", False):
                self.monitor.log_event("setup", f"Pattern {pattern_name} is disabled. Skipping.")
                return
            
            # 基本設定の取得
            llm_config_model = pattern_dict.get("llm_config_model")
            embedding_config_model = pattern_dict.get("embedding_config_model")
            tokenizer_config_model = pattern_dict.get("tokenizer_config_model")
            storage_config = pattern_dict.get("storage_config", {})
            indexing_patterns = pattern_dict.get("indexing_pattern", [])
            data_source = pattern_dict.get("data_source", {})
            
            # コンポーネントのセットアップ
            llm = self._setup_llm(llm_config_model)
            embedding, dim = self._setup_embedding(embedding_config_model)
            tokenizer = self._setup_tokenizer(tokenizer_config_model)
            
            Settings.llm = llm
            Settings.embed_model = embedding
            Settings.tokenizer = tokenizer
            
            # データベースとストレージの設定
            self._setup_database_manager()
            self._setup_storage_context_manager(self.db_manager)
            
            # Milvusスキーマの設定
            context_name = storage_config.get("context_name")
            schema = self._get_milvus_schema(context_name, dim)
            storage_config["vector_store"]["schema"] = schema
            storage_config["vector_store"]["dim"] = dim
            
            # ストレージコンテキストの作成
            storage_context = self._setup_storage_context(storage_config, self.storage_context_manager)
            
            # ファイルパスの取得
            self.monitor.log_event("loading", "Getting file paths...")
            file_paths = self._get_file_paths_from_data_source(data_source)
            
            if not file_paths:
                self.monitor.log_event("loading", "No files found. Skipping.")
                return
            
            self.monitor.log_event("loading", f"Found {len(file_paths)} files to process")

            # 各インデックスパターンに対してインデックスを構築
            for idx_pattern in indexing_patterns:
                index_type = idx_pattern.get("type")
                chunking_config_model = idx_pattern.get("chunking_config_model")
                extractor_pattern = idx_pattern.get("extractor_pattern")
                
                self.monitor.log_event("indexing", f"Building {index_type} index...")
                
                # チャンカーとエクストラクタの設定
                chunker = self._setup_chunker(chunking_config_model)
                extractors = self._setup_extractors(extractor_pattern)
                
                # インデックスビルダーの作成
                index_builder = self._setup_indexbuilder(index_type, storage_context)
                
                # IngestionPipelineの作成
                pipeline = IngestionPipeline(
                    transformations=[chunker] + extractors
                )
                
                # ファイルごとにドキュメントを読み込み、ノードを生成してインデックスに追加
                loader = DocumentLoader()
                total_nodes = 0
                total_documents = 0
                successful_files = 0
                failed_files = []
                
                for file_path in file_paths:
                    try:
                        # 1ファイルずつ読み込み
                        self.monitor.log_event("loading", f"Processing file: {file_path}")
                        documents = loader.load_from_file(file_path)

                        if not documents:
                            self.monitor.log_event("loading", f"No documents loaded from {file_path}")
                            continue

                        for doc in documents:
                            doc.metadata["term_name"] = file_path.stem
                        
                        # ドキュメントのメタデータをログに記録
                        self.monitor.log_document_metadata(str(file_path), documents)
                        
                        # ノードの生成
                        nodes = pipeline.run(documents=documents, show_progress=True)
                        
                        # ノードのメタデータをログに記録
                        if nodes:
                            self.monitor.log_node_metadata(str(file_path), nodes)
                        
                        if nodes:
                            # インデックスに追加
                            index_builder.build_from_nodes(nodes)
                            total_nodes += len(nodes)
                            total_documents += len(documents)
                            successful_files += 1
                            self.monitor.log_event("indexing", 
                                f"Added {len(nodes)} nodes from {Path(file_path).name} (Total: {total_nodes} nodes from {successful_files} files)",
                                )
                        else:
                            self.monitor.log_event("loading", f"No nodes created from {file_path}")
                            
                    except Exception as e:
                        self.monitor.log_event("error", f"Failed to process file {file_path}: {e}")
                        failed_files.append({"file": file_path, "error": str(e)})
                        continue
                
                # インデックスの取得
                index = index_builder.get_index()
                self.monitor.log_event("indexing", 
                    f"Built {index_type} index successfully: "
                    f"{total_nodes} nodes from {successful_files}/{len(file_paths)} files")
                
                if failed_files:
                    self.monitor.log_event("error", f"{len(failed_files)} files failed to process")

                # 結果の保存
                phase_result = {
                    "index_type": index_type,
                    "num_nodes": total_nodes,
                    "num_documents": total_documents,
                    "total_files": len(file_paths),
                    "successful_files": successful_files,
                    "failed_files": failed_files,
                    "chunking_config": chunking_config_model,
                    "extractor_pattern": extractor_pattern,
                }
                self._save_phase_result(
                    self.monitor.get_current_test_dir(),
                    f"index_{index_type}",
                    phase_result
                )
            
            self.monitor.log_event("complete", f"Pattern {pattern_name} completed successfully")
            self.monitor.end_test(True, phase_result)
            
        except Exception as e:
            self.monitor.log_event("error", f"Pattern execution failed: {e}")
            self.monitor.end_test(False, phase_result)
            raise
    
    def run_all_enabled_patterns(self):
        """有効なすべてのパターンを実行"""
        enabled_patterns = self.test_pattern_manager.get_enabled_test_patterns("store_domain")
        
        self.monitor.log_event("setup", f"Found {len(enabled_patterns)} enabled patterns")
        
        for pattern_name in enabled_patterns:
            try:
                self.monitor.log_event("setup", f"Running pattern: {pattern_name}")
                self.run_pattern(pattern_name)
            except Exception as e:
                self.monitor.log_event("error", f"Pattern {pattern_name} failed: {e}")
                continue


class QueryDomainKnowledgeRunner(TestRunnerBase):
    """保存されたドメイン知識をRAGで検索するテストランナー"""
    
    def _get_milvus_schema(self, context_name: str, dim: int):
        """コンテキスト名に応じたMilvusスキーマを取得"""
        if "glossary" in context_name.lower():
            return GlossaryTerm.get_milvus_schema(dim)
        elif "tech_column" in context_name.lower():
            return TechColumnTerm.get_milvus_schema(dim)
        else:
            raise ValueError(f"Unknown context name for schema: {context_name}")
    
    def _create_post_processors(self, node_postprocessors_config: List[Dict[str, Any]]):
        """ポストプロセッサーのリストを作成"""
        post_processors = []
        
        for pp_config in node_postprocessors_config:
            pp_type = pp_config.get("type")
            
            if pp_type == "similarity_cutoff":
                from llama_index.core.postprocessor import SimilarityPostprocessor
                cutoff = pp_config.get("cutoff", 0.7)
                post_processors.append(SimilarityPostprocessor(similarity_cutoff=cutoff))
                self.monitor.log_event("setup", f"Added SimilarityPostprocessor with cutoff={cutoff}")
                
            elif pp_type == "keyword_filter":
                from llama_index.core.postprocessor import KeywordNodePostprocessor
                required_keywords = pp_config.get("required_keywords", [])
                exclude_keywords = pp_config.get("exclude_keywords", [])
                post_processors.append(
                    KeywordNodePostprocessor(
                        required_keywords=required_keywords,
                        exclude_keywords=exclude_keywords
                    )
                )
                self.monitor.log_event("setup", f"Added KeywordNodePostprocessor")
                
            elif pp_type == "rerank":
                # リランカーの実装が必要な場合
                top_n = pp_config.get("top_n", 5)
                self.monitor.log_event("setup", f"Rerank postprocessor requested (top_n={top_n})")
                # 実装例: CohereRerank, SentenceTransformerRerank など
                
            else:
                self.monitor.log_event("warning", f"Unknown postprocessor type: {pp_type}")
        
        return post_processors
    
    def run_pattern(self, pattern_name: str, query: str):
        """
        指定されたパターンに従ってクエリを実行
        
        Args:
            pattern_name: テストパターン名
            query: 検索クエリ文字列
        """
        try:
            TemplatePromptSettings.initialize(self.config_manager.get_template_prompts())
            
            pattern_dict = self.test_pattern_manager.get_store_domain_test_pattern(pattern_name)
            
            self.monitor.start_test(f"{pattern_name}_query", pattern_dict)
            
            # パターン設定の読み込み
            self.monitor.log_event("setup", f"Loading pattern: {pattern_name}")
            
            if not pattern_dict:
                raise ValueError(f"Pattern not found: {pattern_name}")
            
            if not pattern_dict.get("enabled", False):
                self.monitor.log_event("setup", f"Pattern {pattern_name} is disabled. Skipping.")
                return
            
            # 基本設定の取得
            llm_config_model = pattern_dict.get("llm_config_model")
            embedding_config_model = pattern_dict.get("embedding_config_model")
            tokenizer_config_model = pattern_dict.get("tokenizer_config_model")
            storage_config = pattern_dict.get("storage_config", {})
            indexing_patterns = pattern_dict.get("indexing_pattern", [])
            query_config = pattern_dict.get("query_config", {})
            
            # コンポーネントのセットアップ
            llm = self._setup_llm(llm_config_model)
            embedding, dim = self._setup_embedding(embedding_config_model)
            tokenizer = self._setup_tokenizer(tokenizer_config_model)
            
            Settings.llm = llm
            Settings.embed_model = embedding
            Settings.tokenizer = tokenizer
            
            # データベースとストレージの設定
            self._setup_database_manager()
            self._setup_storage_context_manager(self.db_manager)
            
            # Milvusスキーマの設定
            context_name = storage_config.get("context_name")
            schema = self._get_milvus_schema(context_name, dim)
            storage_config["vector_store"]["schema"] = schema
            storage_config["vector_store"]["dim"] = dim
            
            # ストレージコンテキストの作成（既存のインデックスをロード）
            storage_context = self._setup_storage_context(storage_config, self.storage_context_manager)
            
            # query_configからパラメータを取得
            similarity_top_k = query_config.get("similarity_top_k", 10)
            response_mode = query_config.get("response_mode", "compact")
            streaming = query_config.get("streaming", False)
            node_postprocessors_config = query_config.get("node_postprocessors", [])
            
            # ポストプロセッサーの作成
            node_postprocessors = self._create_post_processors(node_postprocessors_config)
            
            query_results = []
            
            # 各インデックスタイプに対してクエリを実行
            for idx_pattern in indexing_patterns:
                index_type = idx_pattern.get("type")
                
                self.monitor.log_event("querying", f"Loading {index_type} index...")
                
                # インデックスビルダーから既存のインデックスをロード
                index_builder = self._setup_indexbuilder(index_type, storage_context)
                index = index_builder.get_index()
                
                self.monitor.log_event("querying", f"Creating retriever for {index_type} index...")
                
                # Retrieverの作成
                retriever = RetrieverFactory.create(
                    retriever_type=index_type,
                    index=index,
                    similarity_top_k=similarity_top_k
                )
                
                self.monitor.log_event("querying", f"Creating query engine for {index_type} index...")
                
                # QueryEngineの作成
                from factories.response_synthesizer_factory import ResponseMode
                
                # response_modeを適切なResponseMode enumに変換
                response_mode_enum = getattr(ResponseMode, response_mode.upper(), ResponseMode.COMPACT)
                
                query_engine = QueryEngineFactory.create_retriever_query_engine(
                    index=index,
                    retriever=retriever,
                    response_mode=response_mode_enum
                )
                
                # ポストプロセッサーを設定
                if node_postprocessors:
                    query_engine._node_postprocessors = node_postprocessors
                
                self.monitor.log_event("querying", f"Executing query on {index_type} index: {query}")
                
                # クエリの実行
                if streaming:
                    response = query_engine.query(query)
                    # ストリーミングレスポンスの処理
                    full_response = ""
                    if hasattr(response, 'response_gen'):
                        for token in response.response_gen:
                            full_response += token
                    else:
                        full_response = str(response)
                    
                    result = {
                        "index_type": index_type,
                        "query": query,
                        "response": full_response,
                        "source_nodes": [
                            {
                                "node_id": node.node.node_id,
                                "score": node.score,
                                "text": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                                "metadata": node.node.metadata
                            }
                            for node in response.source_nodes
                        ] if hasattr(response, 'source_nodes') else []
                    }
                else:
                    response = query_engine.query(query)
                    
                    result = {
                        "index_type": index_type,
                        "query": query,
                        "response": str(response),
                        "source_nodes": [
                            {
                                "node_id": node.node.node_id,
                                "score": node.score,
                                "text": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                                "metadata": node.node.metadata
                            }
                            for node in response.source_nodes
                        ] if hasattr(response, 'source_nodes') else []
                    }
                
                query_results.append(result)
                
                self.monitor.log_event("querying", 
                    f"Query completed for {index_type} index. "
                    f"Retrieved {len(result['source_nodes'])} source nodes")
            
            # 結果の保存
            phase_result = {
                "query": query,
                "query_config": query_config,
                "results": query_results
            }
            
            self._save_phase_result(
                self.monitor.get_current_test_dir(),
                "query_results",
                phase_result
            )
            
            self.monitor.log_event("complete", f"Query execution completed for pattern {pattern_name}")
            self.monitor.end_test(True, phase_result)
            
            return query_results
            
        except Exception as e:
            self.monitor.log_event("error", f"Query execution failed: {e}")
            self.monitor.end_test(False, {"error": str(e)})
            raise
    
    def run_multiple_queries(self, pattern_name: str, queries: List[str]):
        """
        複数のクエリを実行
        
        Args:
            pattern_name: テストパターン名
            queries: 検索クエリ文字列のリスト
        """
        all_results = []
        
        for query in queries:
            try:
                self.monitor.log_event("querying", f"Processing query: {query}")
                results = self.run_pattern(pattern_name, query)
                all_results.append({
                    "query": query,
                    "results": results
                })
            except Exception as e:
                self.monitor.log_event("error", f"Query '{query}' failed: {e}")
                all_results.append({
                    "query": query,
                    "error": str(e)
                })
                continue
        
        return all_results


if __name__ == "__main__":
    # パスの設定
    current_dir = Path(__file__).parent
    config_dir = current_dir / "config"
    test_dir = current_dir / "tests"
    result_dir = current_dir / "results" / "store_domain_knowledge_tests"
    
    # テストランナーの実行
    runner = StoreDomainKnowledgeRunner(
        config_dir=str(config_dir),
        test_dir=str(test_dir),
        result_dir=str(result_dir)
    )
    
    runner.run_pattern('store_glossary_basic')
