from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer, ResponseMode
from factories.prompt_template import *

class ResponseSynthesizerFactory:
    @staticmethod
    def create(response_mode: ResponseMode) -> BaseSynthesizer:
        synthesizer = get_response_synthesizer(
            text_qa_template=DEFAULT_TEXT_QA_PROMPT_TMPL,
            refine_template=DEFAULT_REFINE_PROMPT_TMPL,
            simple_template=DEFAULT_SIMPLE_INPUT_TMPL
            summary_template=DEFAULT_SUMMARY_PROMPT_TMPL,
            response_mode=response_mode,
            use_async=False
        )
        return synthesizer