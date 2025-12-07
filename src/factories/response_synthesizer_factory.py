from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer, ResponseMode
from .template_prompts import TemplatePromptSettings

class ResponseSynthesizerFactory:
    @staticmethod
    def get(llm, prompthelper, response_mode: ResponseMode) -> BaseSynthesizer:
        synthesizer = get_response_synthesizer(
            llm=llm,
            prompt_helper=prompthelper,
            text_qa_template=TemplatePromptSettings.JP_TEXT_QA_PROMPT,
            refine_template=TemplatePromptSettings.JP_REFINE_PROMPT,
            simple_template=TemplatePromptSettings.JP_SIMPLE_INPUT_PROMPT,
            summary_template=TemplatePromptSettings.JP_TREE_SUMMARIZE_PROMPT,
            response_mode=response_mode,
        )
        return synthesizer
