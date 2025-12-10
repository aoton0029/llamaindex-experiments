from typing import Optional, Callable, List
from llama_index.core.indices.prompt_helper import PromptHelper
import tiktoken
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class PromptHelperFactory:
    @staticmethod
    def create_jp(
        tokenizer,
        context_window: int = 2048,
        num_output: int = 512,
        chunk_overlap_ratio: float = 0.1,
        chunk_size_limit: Optional[int] = None,
        separator = "。"
    ):
        try:
            def tokenizer_fn(text: str) -> List:
                return tokenizer.encode(text)

            helper = PromptHelper(
                context_window=context_window,
                num_output=num_output,
                chunk_overlap_ratio=chunk_overlap_ratio,
                chunk_size_limit=chunk_size_limit,
                tokenizer=tokenizer_fn,
                separator=separator,
            )
            logger.info(
                f"PromptHelperを作成: context_window={context_window}, "
                f"num_output={num_output}, separator='{separator}'"
            )
            return helper
        except Exception as e:
            logger.error(f"PromptHelper作成エラー: {e}")
            raise