from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
    MetadataReplacementPostProcessor,
    EmbeddingRecencyPostprocessor,
    PrevNextNodePostprocessor
)

class PostprocessorFactory:
    def __init__(self):
        self.postprocessors = {
            "similarity": SimilarityPostprocessor,
            "keyword": KeywordNodePostprocessor,
            "metadata_replacement": MetadataReplacementPostProcessor,
            "embedding_recency": EmbeddingRecencyPostprocessor,
            "prev_next": PrevNextNodePostprocessor,
        }
    
    def create_postprocessor(self, postprocessor_type: str, **kwargs) -> BaseNodePostprocessor:
        if postprocessor_type in self.postprocessors:
            return self.postprocessors[postprocessor_type](**kwargs)
        else:
            raise ValueError(f"Unknown postprocessor type: {postprocessor_type}")