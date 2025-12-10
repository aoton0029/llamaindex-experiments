import logging
from llama_index.core.llms.llm import BaseLLM
from llama_index.core.embeddings import BaseEmbedding


logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create(backend: str, model_name: str, base_url: str, **kwargs) -> BaseLLM:
        if backend == "ollama":
            from llama_index.llms.ollama import Ollama
            return Ollama(
                model=model_name, 
                base_url=base_url,
                **kwargs)
        elif backend == "vllm":
            from llama_index.llms.openai_like import OpenAILike
            return OpenAILike(
                model=model_name, 
                api_base=base_url,
                **kwargs)
        raise ValueError(f"Unsupported LLM backend: {backend}")


class EmbeddingFactory:
    @staticmethod
    def create(backend: str, model_name: str, base_url: str, **kwargs) -> BaseEmbedding:
        if backend == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            return OllamaEmbedding(
                model_name=model_name,
                base_url=base_url,
                **kwargs)
        elif backend == "vllm":
            from llama_index.embeddings.openai_like import OpenAILikeEmbedding
            return OpenAILikeEmbedding(
                model_name=model_name, 
                api_base=base_url, 
                **kwargs)

        raise ValueError(f"Unsupported Embedding backend: {backend}")
