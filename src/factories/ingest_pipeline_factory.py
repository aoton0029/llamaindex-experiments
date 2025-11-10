from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)

class PipelineFactory:
    @staticmethod
    def create(ingest_type:str, **kwargs) -> IngestionPipeline:
        if ingest_type == "default":
            return PipelineFactory.create_default_pipeline()
        else:
            raise ValueError(f"未知のパイプラインタイプ: {ingest_type}")
    
    @staticmethod
    def create_default_pipeline() -> IngestionPipeline:
        
        pipeline = IngestionPipeline(
            name="default",
            project_name=,
            transformations=,
            readers,
            vector_store,
            docstore,
            docstore_strategy,
            
        )
        return pipeline