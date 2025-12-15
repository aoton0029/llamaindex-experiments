from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer, ResponseMode
from llama_index.core.prompts import PromptTemplate, SelectorPromptTemplate
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate, PromptType
from abc import ABC, abstractmethod

class ResponseSynthesizerFactoryBase(ABC):
    def is_chat_model(llm: BaseLLM) -> bool:
        return llm.metadata.is_chat_model


    @abstractmethod
    def _get_text_qa_system_prompt_tmpl(self) -> str:
        return (
                "You are an expert Q&A system that is trusted around the world.\n"
                "Always answer the query using the provided context information, "
                "and not prior knowledge.\n"
                "Some rules to follow:\n"
                "1. Never directly reference the given context in your answer.\n"
                "2. Avoid statements like 'Based on the context, ...' or "
                "'The context information ...' or anything along "
                "those lines."
        )
    
    @abstractmethod
    def _get_text_qa_user_prompt_tmpl(self) -> str:
        return (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

    @abstractmethod
    def _get_tree_summarize_prompt_tmpl(self) -> str:
        return (
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        
    @abstractmethod
    def _get_tree_summarize_user_prompt_tmpl(self) -> str:
        return (
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
    

    @abstractmethod
    def _get_refine_user_prompt_tmpl(self) -> str:
        return (
                "You are an expert Q&A system that strictly operates in two modes "
                "when refining existing answers:\n"
                "1. **Rewrite** an original answer using the new context.\n"
                "2. **Repeat** the original answer if the new context isn't useful.\n"
                "Never reference the original answer or context directly in your answer.\n"
                "When in doubt, just repeat the original answer.\n"
                "New Context: {context_msg}\n"
                "Query: {query_str}\n"
                "Original Answer: {existing_answer}\n"
                "New Answer: "
            )
    @abstractmethod
    def _get_text_qa_prompt_tmpl(self) -> str:
        return (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
    
    @abstractmethod
    def _get_refine_prompt_tmpl(self) -> str:
        return (
            "The original query is as follows: {query_str}\n"
            "We have provided an existing answer: {existing_answer}\n"
            "We have the opportunity to refine the existing answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the query. "
            "If the context isn't useful, return the original answer.\n"
            "Refined Answer: "
        )
    
    @abstractmethod
    def _get_refine_table_context_prompt_tmpl(self) -> str:
        return (
            "We have provided a table schema below. "
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "We have also provided some context information below. "
            "{context_msg}\n"
            "---------------------\n"
            "Given the context information and the table schema, "
            "refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer."
        )
    

    def _get_qa_system_prompt(self) -> ChatMessage:
        return ChatMessage(
            content=self._get_text_qa_system_prompt_tmpl(),
            role=MessageRole.SYSTEM,
        )
    

    def _get_chat_text_qa_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(message_templates=[
            self._get_qa_system_prompt(),
            ChatMessage(
                content=self._get_text_qa_user_prompt_tmpl(),
                role=MessageRole.USER,
            ),
        ])

    
    def _get_tree_summarize_prompt_tmpl_msgs(self):
        return [
            self._get_qa_system_prompt(),
            ChatMessage(
                content=self._get_tree_summarize_user_prompt_tmpl(),
                role=MessageRole.USER,
            ),
        ]

    
    def _get_tree_summarize_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(
            message_templates=self._get_tree_summarize_prompt_tmpl_msgs()
        )

    
    def _get_chat_refine_prompt_tmpl_msgs(self) -> list[ChatMessage]:
        return [
            ChatMessage(
                content=self._get_refine_user_prompt_tmpl(),
                role=MessageRole.USER,
            ),
        ]
    
    
    def _get_refine_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(
            message_templates=self._get_chat_refine_prompt_tmpl_msgs()
        )


    def _get_refine_table_context_tmpl_msgs(self) -> list[ChatMessage]:
        return [
            ChatMessage(content="{query_str}", role=MessageRole.USER),
            ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
            ChatMessage(
                content=self._get_refine_table_context_prompt_tmpl(),
                role=MessageRole.USER,
            ),
        ]
    
    
    def _get_chat_refine_table_context_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(
            message_templates=self._get_refine_table_context_tmpl_msgs()
        )
    
    
    def _get_text_qa_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            template=self._get_text_qa_prompt_tmpl(), 
            prompt_type=PromptType.QUESTION_ANSWER,
        )


    def _get_refine_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            template=self._get_refine_prompt_tmpl(), 
            prompt_type=PromptType.REFINE,
        )

    
    def _get_tree_summarize_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            template=self._get_tree_summarize_prompt_tmpl(), 
            prompt_type=PromptType.SUMMARY,
        )
    
    
    def _get_text_qa_prompt_sel(self) -> SelectorPromptTemplate:
        return SelectorPromptTemplate(
            default_template=self._get_text_qa_prompt(),
            conditionals=[(self.is_chat_model, self._get_chat_text_qa_prompt())],
        )

    
    def _get_refine_prompt_sel(self) -> SelectorPromptTemplate:
        return SelectorPromptTemplate(
            default_template=self._get_refine_prompt(),
            conditionals=[(self.is_chat_model, ChatPromptTemplate(message_templates=self._get_chat_refine_prompt_tmpl_msgs()))],
        )

    
    def _get_simple_input_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            template="{query_str}", prompt_type=PromptType.SIMPLE_INPUT
        )

    
    def _get_tree_summarize_prompt_sel(self) -> SelectorPromptTemplate:
        return SelectorPromptTemplate(
            default_template=self._get_tree_summarize_prompt(),
            conditionals=[
                (self.is_chat_model, ChatPromptTemplate(message_templates=self._get_tree_summarize_prompt_tmpl_msgs()))
            ],
        )

    
    def create_synthesizer(self, llm: BaseLLM) -> BaseSynthesizer:
        return get_response_synthesizer(
            llm=llm,
            text_qa_template=self._get_text_qa_prompt_sel(),
            refine_template=self._get_refine_prompt_sel(),
            simple_input_template=self._get_simple_input_prompt(),
            tree_summarize_template=self._get_tree_summarize_prompt_sel(),
            response_mode=ResponseMode.COMPACT
        )

class TranscriptionResponseSynthesizerFactory(ResponseSynthesizerFactoryBase):
    def _get_text_qa_system_prompt_tmpl(self) -> str:
        return (
            "You are an expert Q&A system that is trusted around the world.\n"
            "Always answer the query using the provided context information, "
            "and not prior knowledge.\n"
            "Some rules to follow:\n"
            "1. Never directly reference the given context in your answer.\n"
            "2. Avoid statements like 'Based on the context, ...' or "
            "'The context information ...' or anything along "
            "those lines."
        )
    
    def _get_text_qa_user_prompt_tmpl(self) -> str:
        return (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

    def _get_tree_summarize_prompt_tmpl(self) -> str:
        return (
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        
    def _get_tree_summarize_user_prompt_tmpl(self) -> str:
        return (
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
    

    def _get_refine_user_prompt_tmpl(self) -> str:
        return (
                "You are an expert Q&A system that strictly operates in two modes "
                "when refining existing answers:\n"
                "1. **Rewrite** an original answer using the new context.\n"
                "2. **Repeat** the original answer if the new context isn't useful.\n"
                "Never reference the original answer or context directly in your answer.\n"
                "When in doubt, just repeat the original answer.\n"
                "New Context: {context_msg}\n"
                "Query: {query_str}\n"
                "Original Answer: {existing_answer}\n"
                "New Answer: "
            )
    def _get_text_qa_prompt_tmpl(self) -> str:
        return (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
    
    def _get_refine_prompt_tmpl(self) -> str:
        return (
            "The original query is as follows: {query_str}\n"
            "We have provided an existing answer: {existing_answer}\n"
            "We have the opportunity to refine the existing answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the query. "
            "If the context isn't useful, return the original answer.\n"
            "Refined Answer: "
        )
    
    def _get_refine_table_context_prompt_tmpl(self) -> str:
        return (
            "We have provided a table schema below. "
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "We have also provided some context information below. "
            "{context_msg}\n"
            "---------------------\n"
            "Given the context information and the table schema, "
            "refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer."
        )
    
    