from pathlib import Path
from llama_index.core import Settings
from test_runner_base import TestRunnerBase
from models import (
    GlossaryTerm,
    TechColumnTerm,
    PdfDocumentVector,
)
from factories import (
    ExtractorFactory,
    DocumentLoader,
    IndexBuilderFactory
)
from factories.template_prompts import TemplatePromptSettings
from db import (
    DatabaseManager,
    DatabaseConfig,
    StorageContextConfig,
    StorageContextManager,
)
import utils


class StoreGlossaryRunner(TestRunnerBase):
    terms_dir = Path(".") / "datas" / "glossary" / "terms"
    milvus_collection_name = "glossary_terms"
    mongodb_namespace = "glossary_db"
    index_namespace = "glossary_index"
    
    def run_test(name:str):
        pass

    def run(self):
        """
        - glossary_dir 内の全ての .md ファイルを読み込み、データベースに保存する。
        - StorageContext を使用して、各用語のタイトルと内容を格納する。
            - vector_store, docstore, index_store, 
        - SummaryIndex, VectorStoreIndex を作成し、データベースに保存する。
        """
        # Step 1: Setup LLM, Embedding, Tokenizer
        TemplatePromptSettings.initialize(self.config_manager.get_template_prompts())
        llm_config = self.config_manager.get_llm_config("vllm_elyza_8b_awq")
        embedding_config = self.config_manager.get_embedding_config("ollama_qwen3")
        tokenizer_config = self.config_manager.get_tokenizer_config("hf_elyza_8b_awq")
        
        Settings.llm = self._setup_llm(llm_config)
        Settings.embed_model, dim = self._setup_embedding(embedding_config)
        Settings.tokenizer = self._setup_tokenizer(tokenizer_config)

        # Step 2: Setup Database and StorageContext
        db_config = DatabaseConfig()
        db_manager = DatabaseManager(config=db_config)
        db_manager.connect_all()
        storage_manager = StorageContextManager(db_manager=db_manager)
        storage_config = StorageContextConfig.from_dict({
            "context_name": "glossary",
            "vector_store": {
                "collection_name": self.milvus_collection_name,
                "dim": dim,
                "schema": GlossaryTerm.get_milvus_schema(dim),
                "metric_type": "IP",
                "index_type": "HNSW",
                "additional_params": {}
            },
            "docstore": {
                "namespace": self.mongodb_namespace,
                "collection_name": "glossary_documents"
            },
            "index_store": {
                "namespace": self.index_namespace,
                "collection_suffix": "glossary"
            }
        })
        
        storage_context = storage_manager.create_storage_context(storage_config)
        storage_manager.drop_storage_context(
            "glossary",
            self.milvus_collection_name,
            self.mongodb_namespace,
            self.index_namespace,
        )
        storage_context = storage_manager.create_storage_context(storage_config)
        self.monitor.log_event("setup", "Storage context created successfully")

        # Step 3: Load documents from glossary terms directory
        document_loader = DocumentLoader()
        all_documents = []
        
        for md_file in self.terms_dir.glob("*.md"):
            try:
                documents = document_loader.load_from_file(str(md_file))
                # Add metadata to documents
                for doc in documents:
                    doc.metadata["term_name"] = md_file.stem
                    doc.metadata["file_path"] = str(md_file)
                all_documents.extend(documents)
                self.monitor.log_event("loading", f"Loaded: {md_file.name}")
                break
            except Exception as e:
                self.monitor.log_event("error", f"Failed to load {md_file.name}: {e}")
        
        self.monitor.log_event("loading", f"Total documents loaded: {len(all_documents)}")

        # Step 4: Setup chunker and split documents into nodes
        chunking_config = {
            "type": "sentence",
            "kwargs": {
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "separator": "。"
            }
        }
        chunker = self._setup_chunker(chunking_config)
        nodes = chunker.get_nodes_from_documents(all_documents, show_progress=True)
        self.monitor.log_event("chunking", f"Total nodes created: {len(nodes)}")

        # Step 5: Setup extractors for metadata enrichment
        extractor_configs = [
            {
                "type": "title",
                "kwargs": {
                    "nodes": 5
                }
            },
            {
                "type": "summary",
                "kwargs": {
                }
            },
            {
                "type": "keyword",
                "kwargs": {
                    "keywords": 5,
                }
            }
        ]

        extractors = self._setup_extractors(extractor_configs)
        
        # Apply extractors to nodes
        for extractor in extractors:
            nodes = extractor.process_nodes(nodes, show_progress=True)
        self.monitor.log_event("extraction", "Metadata extraction completed")

        # Step 6: Build VectorStoreIndex
        vector_index_builder = self._setup_indexbuilder(
            indexing_type="vector_store",
            storage_context=storage_context
        )
        vector_index = vector_index_builder.build_from_nodes(nodes)
        self.monitor.log_event("indexing", "VectorStoreIndex created successfully")

        # Step 7: Build SummaryIndex
        summary_index_builder = self._setup_indexbuilder(
            indexing_type="summary",
            storage_context=storage_context
        )
        summary_index = summary_index_builder.build_from_nodes(nodes)
        self.monitor.log_event("indexing", "SummaryIndex created successfully")

        # Step 8: Save results
        self._save_phase_result(
            experiment_dir=Path(self.result_dir),
            phase_name="glossary_indexing",
            data={
                "total_documents": len(all_documents),
                "total_nodes": len(nodes),
                "vector_index_id": vector_index.index_id,
                "summary_index_id": summary_index.index_id,
                "storage_config": storage_config.to_dict()
            }
        )
        
        self.monitor.log_event("completion", "Glossary indexing completed successfully")
        

class StoreTechRunner(TestRunnerBase):
    terms_dir = Path(".") / "datas" / "tech_column" / "terms"

    def run(self):
        return super().run()


if __name__ == "__main__":
    # パスの設定
    current_dir = Path(__file__).parent
    config_dir = current_dir / "config"
    data_dir = current_dir.parent / "datas" / "pdf"
    result_dir = current_dir / "results" / "store_domain_knowledge_tests"
    
    # テストランナーの実行
    store_glossary_runner = StoreGlossaryRunner(
        config_dir=str(config_dir),
        data_dir=str(data_dir),
        result_dir=str(result_dir)
    )
    store_glossary_runner.run()

    # store_tech_runner = StoreTechRunner()
    # store_tech_runner.run()
