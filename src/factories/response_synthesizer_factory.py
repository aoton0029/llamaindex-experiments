from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer, ResponseMode
from .template_prompts import *

class ResponseSynthesizerFactory:
    @staticmethod
    def get(response_mode: ResponseMode) -> BaseSynthesizer:
        synthesizer = get_response_synthesizer(
            text_qa_template=DEFAULT_TEXT_QA_PROMPT_SEL,
            refine_template=DEFAULT_REFINE_PROMPT_SEL,
            simple_template=DEFAULT_SIMPLE_INPUT_PROMPT,
            summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
            response_mode=response_mode,
            use_async=False
        )
        return synthesizer
