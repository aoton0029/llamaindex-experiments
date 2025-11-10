
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

############################################
# Tree
############################################

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "Write a summary of the following. Try to use only the "
    "information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)

DEFAULT_SUMMARY_PROMPT = PromptTemplate(
    DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
)

# insert prompts
DEFAULT_INSERT_PROMPT_TMPL = (
    "Context information is below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, here is a new piece of "
    "information: {new_chunk_text}\n"
    "Answer with the number corresponding to the summary that should be updated. "
    "The answer should be the number corresponding to the "
    "summary that is most relevant to the question.\n"
)
DEFAULT_INSERT_PROMPT = PromptTemplate(
    DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
)


# # single choice
DEFAULT_QUERY_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
    "Provide choice in the following format: 'ANSWER: <number>' and explain why "
    "this summary was selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT = PromptTemplate(
    DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
)

# multiple choice
DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "(no more than {branching_factor}, ranked by most relevant to least) that "
    "are most relevant to the question: '{query_str}'\n"
    "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
    "these summaries were selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT_MULTIPLE = PromptTemplate(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
)


DEFAULT_REFINE_PROMPT_TMPL = (
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
DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

DEFAULT_TREE_SUMMARIZE_TMPL = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)


############################################
# Keyword Table
############################################

DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Some text is provided below. Given the text, extract up to {max_keywords} "
    "keywords from the text. Avoid stopwords."
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


# NOTE: the keyword extraction for queries can be the same as
# the one used to build the index, but here we tune it to see if performance is better.
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


# single select
DEFAULT_SINGLE_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
)

# multiple select
DEFAULT_MULTI_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "(no more than {max_outputs}, but only select what is needed) that "
    "are most relevant to the question: '{query_str}'\n"
)

############################################
# Simple Input
############################################

DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
DEFAULT_SIMPLE_INPUT_PROMPT = PromptTemplate(
    DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
)