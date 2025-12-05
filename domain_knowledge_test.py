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
    ToolFactory,
    ResponseSynthesizerFactory,
    IndexMetadataExtractor,
    QueryEngineFactory,
    PostProcessorFactory
)
from factories.template_prompts import TemplatePromptSettings


class DomainKnowledgeTestRunner(TestRunnerBase):
    """ドメイン知識をストレージに保存し、RAGクエリを実行するテストランナー"""
    
    def _get_milvus_schema(self, context_name: str, dim: int):
        """コンテキスト名に応じたMilvusスキーマを取得"""
        if "glossary" in context_name.lower():
            return GlossaryTerm.get_milvus_schema(dim)
        elif "tech_column" in context_name.lower():
            return TechColumnTerm.get_milvus_schema(dim)
        else:
            raise ValueError(f"Unknown context name for schema: {context_name}")
    
    def run_indexing_phase(self, pattern_name: str, pattern_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        インデックス作成フェーズ: データをストレージに保存
        
        Args:
            pattern_name: テストパターン名
            pattern_dict: テストパターン設定
            
        Returns:
            インデックス情報を含む辞書
        """
        self.monitor.log_event("indexing", "=== Starting Indexing Phase ===")
        
        # 基本設定の取得
        storage_config = pattern_dict.get("storage_config", {})
        indexing_patterns = pattern_dict.get("indexing_pattern", [])
        data_source = pattern_dict.get("data_source", {})
        
        # ストレージコンテキストの作成
        storage_context = self._setup_storage_context(storage_config, self.storage_context_manager, True)
        
        # ファイルパスの取得
        self.monitor.log_event("loading", "Getting file paths...")
        file_paths = self._get_file_paths_from_data_source(data_source)
        
        if not file_paths:
            self.monitor.log_event("loading", "No files found. Skipping.")
            return {}
        
        self.monitor.log_event("loading", f"Found {len(file_paths)} files to process")
        
        # インデックス結果を保存
        indexing_result = {}

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
                        index = index_builder.build_from_nodes(nodes)
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
                    failed_files.append({"file": str(file_path), "error": str(e)})
                    continue
            
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
            
            self.monitor.log_event(f"{index_type}", f"Complete index: {index_type}", phase_result)
            
            # インデックス結果を保存
            indexing_result[index_type] = phase_result
        
        self.monitor.log_event("indexing", "=== Indexing Phase Completed ===")
        return indexing_result
    
    def run_query_phase(self, pattern_name: str, pattern_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        RAGクエリフェーズ: StorageContextから読み込んでRAGクエリを実行
        
        Args:
            pattern_name: テストパターン名
            pattern_dict: テストパターン設定
            
        Returns:
            クエリ結果を含む辞書
        """
        self.monitor.log_event("query", "=== Starting Query Phase ===")
        
        query_pattern = pattern_dict.get("query_pattern", {})
        
        if not query_pattern.get("enabled", False):
            self.monitor.log_event("query", "Query phase is disabled. Skipping.")
            return {}
        
        # 基本設定の取得
        storage_config = pattern_dict.get("storage_config", {})
        indexing_patterns = pattern_dict.get("indexing_pattern", [])
        retriever_configs = query_pattern.get("retriever_configs", [])
        query_engine_type = query_pattern.get("query_engine_type", "retriever")
        response_mode = query_pattern.get("response_mode", "compact")
        post_processors_configs = query_pattern.get("post_processors", [])
        test_queries = query_pattern.get("test_queries", [])
        
        if not test_queries:
            self.monitor.log_event("query", "No test queries found. Skipping.")
            return {}
        
        # StorageContextを再構築
        storage_context = self._setup_storage_context(storage_config, self.storage_context_manager)
        
        # インデックスをStorageContextからロード
        context_name = storage_config.get("context_name")
        indices = self.storage_context_manager.load_indices(context_name)
        self.monitor.log_event("query", f"Loaded index from storage context: {context_name}")
        
        # # ポストプロセッサを作成
        # post_processors = []
        # for pp_config in post_processors_configs:
        #     pp_type = pp_config.get("type")
        #     pp_kwargs = {k: v for k, v in pp_config.items() if k != "type"}
        #     post_processor = PostProcessorFactory.create(pp_type, **pp_kwargs)
        #     post_processors.append(post_processor)
        #     self.monitor.log_event("query", f"Created {pp_type} post processor")
        
        query_engine_tools = []
        for idx, index in enumerate(indices):
            index_query_engine = index.as_query_engine(
                response_synthesizer = ResponseSynthesizerFactory.get(response_mode=response_mode),
                # node_postprocessors = []
            )

            doc_name = "未設定"            
            doc_desc = "未設定"
            
            from llama_index.core import VectorStoreIndex
            if isinstance(index, VectorStoreIndex):
                milvus_client = self.db_manager.get_milvus_client()
                node_ids = ",".join(f"\"{s}\"" for s in index.index_struct.nodes_dict.keys())                
                values = milvus_client.get_field_values(
                    "tech_column_terms", 
                    f"id in [{node_ids}]", 
                    ["id", "term_name", "doc_id"],
                    1)
                
                if values:
                    doc_name = values[0]["term_name"]
                    doc_desc = f"「{doc_name}」に関する質問に答えます"
            else:
                if hasattr(index, 'ref_doc_info') and index.ref_doc_info:
                    first_doc_info = next(iter(index.ref_doc_info.keys()))
                    metadata = index.ref_doc_info[first_doc_info].metadata
                    doc_name = metadata.get('term_name')
                    doc_desc = f"「{doc_name}」に関する専門的な質問に答えます"
                
            print(f"{index.index_id} term_name:{doc_name} desc:{doc_desc}")
            tool = ToolFactory.create_query_engine_tool(
                query_engine=index_query_engine,
                name=doc_name,
                description=doc_desc
            )
            
            query_engine_tools.append(tool)
        
        # クエリエンジンを作成
        query_engine = QueryEngineFactory.create_router_query_engine(
            selector_type="llm_single",
            query_engine_tools=query_engine_tools,
            response_mode=response_mode,
        )

        # # ポストプロセッサを設定
        # if post_processors:
        #     query_engine._node_postprocessors = post_processors
        
        self.monitor.log_event("query", f"Created {query_engine_type} query engine")
        
        query_results = []
        # テストクエリを実行
        for i, query in enumerate(test_queries, 1):
            try:
                self.monitor.log_event("query", f"Executing query {i}/{len(test_queries)}: {query}")
                response = query_engine.query(query)
                
                query_result = {
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
                    ] if hasattr(response, "source_nodes") else []
                }
                
                query_results.append(query_result)
                self.monitor.log_event("query", f"Query {i} completed successfully")
                
            except Exception as e:
                self.monitor.log_event("error", f"Query {i} failed: {e}")
                query_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        # クエリ結果を保存
        query_phase_result = {
            "num_queries": len(test_queries),
            "successful_queries": len([r for r in query_results if "error" not in r]),
            "failed_queries": len([r for r in query_results if "error" in r]),
            "query_results": query_results
        }
        
        self.monitor.log_event("query_phase", query_phase_result)
        self.monitor.log_event("query", "=== Query Phase Completed ===")
        return query_phase_result
    
    def run_pattern(self, pattern_name: str):
        """
        指定されたパターンに従ってドメイン知識をストレージに保存し、RAGクエリを実行
        
        Args:
            pattern_name: テストパターン名
        """
        try: 
            self._setup_callback()

            # 設定の読み込み
            TemplatePromptSettings.initialize(self.config_manager.get_template_prompts())
            pattern_dict = self.test_pattern_manager.get_domain_knowledge_test_pattern(pattern_name)
            
            self.monitor.start_test(
                pattern_name, 
                {
                    "pattern": pattern_dict,
                    "prompt": TemplatePromptSettings.get_templates_info(),
                })

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
            
            # コンポーネントのセットアップ
            llm = self._setup_llm(llm_config_model, "vllm_default")
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
            
            # Phase 1: インデックス作成フェーズ
            # indexing_result = self.run_indexing_phase(pattern_name, pattern_dict)
            
            # Phase 2: RAGクエリフェーズ（StorageContextから独立して実行）
            query_result = self.run_query_phase(pattern_name, pattern_dict)
            
            # 最終結果をまとめる
            final_result = {
                "indexing": None, #indexing_result,
                "query": query_result
            }
            
            self.monitor.log_event("complete", f"Pattern {pattern_name} completed successfully")
            self.monitor.end_test(True, final_result)
            
        except Exception as e:
            self.monitor.log_event("error", f"Pattern execution failed: {e}")
            self.monitor.end_test(False, {"error": str(e)})
            raise

    
    def run_all_enabled_patterns(self):
        """有効なすべてのパターンを実行"""
        pass
        # enabled_patterns = self.test_pattern_manager.get_test_pattern()
        
        # self.monitor.log_event("setup", f"Found {len(enabled_patterns)} enabled patterns")
        
        # for pattern_name in enabled_patterns:
        #     try:
        #         self.monitor.log_event("setup", f"Running pattern: {pattern_name}")
        #         self.run_pattern(pattern_name)
        #     except Exception as e:
        #         self.monitor.log_event("error", f"Pattern {pattern_name} failed: {e}")
        #         continue



if __name__ == "__main__":
    # パスの設定
    current_dir = Path(__file__).parent
    config_dir = current_dir / "config"
    test_dir = current_dir / "tests"
    result_dir = current_dir / "results" / "domain_knowledge_tests"
    
    # テストランナーの実行
    runner = DomainKnowledgeTestRunner(
        config_dir=str(config_dir),
        test_dir=str(test_dir),
        result_dir=str(result_dir)
    )
    
    runner.run_pattern('tech_column_basic')
