
from llama_index.core import Settings
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate

def is_chat_model() -> bool:
    return Settings.llm.metadata.is_chat_model

# Simple Input
DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
DEFAULT_SIMPLE_INPUT_PROMPT = PromptTemplate(
    DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
)

# Text QA
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "あなたは世界中で信頼されている専門のQ&Aシステムです。\n"
        "常に提供されたコンテキスト情報のみを用いて質問に回答し、事前の知識は使用しないでください。\n"
        "従うべきルール:\n"
        "1. 回答内で与えられたコンテキストを直接参照しないでください。\n"
        "2. 「コンテキストに基づいて...」や「コンテキスト情報...」のような表現を避けてください。"
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "以下にコンテキスト情報を示します。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "コンテキスト情報のみを用いて質問に回答してください（事前知識は使用しないでください）。\n"
            "質問: {query_str}\n"
            "回答: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

JP_TEXT_QA_PROMPT_TMPL = (
    "以下にコンテキスト情報を示します。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "コンテキスト情報のみを用いて質問に回答してください（事前知識は使用しないでください）。\n"
    "質問: {query_str}\n"
    "回答: "
)
JP_TEXT_QA_PROMPT = PromptTemplate(
    JP_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)
default_text_qa_conditionals = [(is_chat_model, CHAT_TEXT_QA_PROMPT)]
DEFAULT_TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=JP_TEXT_QA_PROMPT,
    conditionals=default_text_qa_conditionals,
)


# Tree Summarize
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "複数のソースからのコンテキスト情報を以下に示します。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "複数の情報を踏まえて（事前知識は使用せずに）質問に回答してください。\n"
            "質問: {query_str}\n"
            "回答: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)


# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "あなたは既存の回答を洗練する際、厳密に次の2つのモードで動作する専門のQ&Aシステムです：\n"
            "1. 新しいコンテキストを用いて元の回答を**書き直す**。\n"
            "2. 新しいコンテキストが有用でない場合は元の回答を**繰り返す**。\n"
            "回答内で元の回答やコンテキストを直接参照しないでください。\n"
            "迷ったら元の回答を繰り返してください。\n"
            "新しいコンテキスト: {context_msg}\n"
            "質問: {query_str}\n"
            "元の回答: {existing_answer}\n"
            "新しい回答: "
        ),
        role=MessageRole.USER,
    )
]


CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)

DEFAULT_TREE_SUMMARIZE_TMPL = (
    "複数のソースからのコンテキスト情報を以下に示します。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "複数の情報を踏まえて（事前知識は使用せずに）質問に回答してください。\n"
    "質問: {query_str}\n"
    "回答: "
)
DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)
default_tree_summarize_conditionals = [(is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT)]
DEFAULT_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_TREE_SUMMARIZE_PROMPT,
    conditionals=default_tree_summarize_conditionals,
)

# Refine
DEFAULT_REFINE_PROMPT_TMPL = (
    "元の質問は次の通りです: {query_str}\n"
    "既に提供されている回答: {existing_answer}\n"
    "以下の追加コンテキストを使って、必要に応じて既存の回答を改善する機会があります。\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "新しいコンテキストを踏まえて、元の回答をより良く質問に答えるように洗練してください。"
    "コンテキストが有用でない場合は元の回答を返してください。\n"
    "改善された回答: "
)
DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)
default_refine_conditionals = [(is_chat_model, CHAT_REFINE_PROMPT)]
DEFAULT_REFINE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_PROMPT,
    conditionals=default_refine_conditionals,
)

# Refine Table Context
CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content=(
             "以下にテーブルのスキーマを示します。\n"
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "さらにコンテキスト情報を以下に示します。{context_msg}\n"
            "---------------------\n"
            "テーブルのスキーマとコンテキスト情報を用いて元の回答を改善してください。"
            "コンテキストが有用でない場合は元の回答を返してください。"
        ),
        role=MessageRole.USER,
    ),
]
CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
    message_templates=CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
)
DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
     "以下にテーブルのスキーマを示します。\n"
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "さらにコンテキスト情報を以下に示します。{context_msg}\n"
    "---------------------\n"
    "テーブルのスキーマとコンテキスト情報を用いて次のタスクに対する回答を作成してください: {query_str}\n"
    "既に提供されている回答: {existing_answer}\n"
    "新しいコンテキストを踏まえて、元の回答をより良くするように改善してください。"
    "コンテキストが有用でない場合は元の回答を返してください。"
)
DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)
default_refine_table_conditionals = [(is_chat_model, CHAT_REFINE_TABLE_CONTEXT_PROMPT)]
DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_TABLE_CONTEXT_PROMPT,
    conditionals=default_refine_table_conditionals,
)


# single select
DEFAULT_SINGLE_SELECT_PROMPT_TMPL = (
    "以下に候補が番号付きリスト（1〜{num_choices}）で示されています。各項目はサマリーに対応します。\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "上記の選択肢のみを用い、事前の知識は使わずに、質問'{query_str}'に最も関連する選択肢を返してください。\n"
)


# multiple select
DEFAULT_MULTI_SELECT_PROMPT_TMPL = (
    "以下に候補が番号付きリスト（1〜{num_choices}）で示されています。各項目はサマリーに対応します。\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "上記の選択肢のみを用い、事前の知識は使わず、質問'{query_str}'に最も関連する上位の選択肢を返してください（最大で{max_outputs}件まで、必要な分だけ選んでください）。\n"
)


##############################################################
# Extractor Prompt Templates
##############################################################
DEFAULT_TITLE_NODE_TEMPLATE = """コンテキスト: {context_str}\nこの内容に含まれる固有の項目、見出し、またはテーマをまとめて表す短いタイトルを付けてください。\nタイトル: """
DEFAULT_TITLE_COMBINE_TEMPLATE = """{context_str}\n上記の候補タイトルと内容に基づいて、この文書全体を最も包括的に表すタイトルを決定してください。\nタイトル: """

DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
セクションの内容は以下の通りです:
{context_str}

このセクションの主要なトピックと登場する主体（エンティティ）を簡潔に要約してください。

要約: """

DEFAULT_KEYWORD_EXTRACT_TEMPLATE = """\
{context_str}\nこの文書を表す重複のないキーワードを{keywords}個挙げてください。カンマ区切りで記載してください。\nキーワード: """

DEFAULT_QUESTION_GEN_TMPL = """\
コンテキストは以下の通りです:
{context_str}

このコンテキストに基づき、他では見つかりにくい具体的な解答をこの文脈から導ける質問を{num_questions}件生成してください。

周辺の文脈を高次に要約した情報を付け加えても構いません。これらの要約を活用して、このコンテキストが答えられるより適切な質問を作成してください。"""

###############################################################
# Evaluation Prompt Templates
###############################################################
JP_TEXT_QA_PROMPT_TMPL = (
    "コンテキスト情報は以下の通りです。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "上記のコンテキスト情報を使用して、以下の質問に答えてください。\n"
    "回答する際の注意事項:\n"
    "- コンテキストに基づいた正確な回答をしてください\n"
    "- コンテキストに情報がない場合は「提供された情報からは回答できません」と答えてください\n"
    "- 回答は簡潔かつ明確にしてください\n"
    "- 自然な日本語で回答してください\n\n"
    "質問: {query_str}\n"
    "回答: "
)

JP_TEXT_QA_PROMPT = PromptTemplate(
    JP_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

JP_QUESTION_GENERATION_PROMPT = (
    "コンテキスト情報は以下の通りです。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "上記のコンテキスト情報のみを基に、質問を生成してください\n\n"
    "## 質問を生成する際の注意事項\n"
    "1. 質問はコンテキストから直接回答できるものにしてください。\n"
    "2. 質問は具体的で明確なものにしてください\n"
    "3. 質問は以下のような多様性を持たせてください\n"
    "   1. 事実確認の質問（「～は何ですか？」「～はいつですか？」）\n"
    "   2. 理由・説明の質問（「なぜ～ですか？」「どのように～ですか？」）\n"
    "   3. 比較の質問（「～と～の違いは何ですか？」）\n"
    "   4. 評価の質問（「～の利点は何ですか？」）\n\n"
    "4. 各質問は独立した行に記述してください\n"
    "5. 質問文は自然な日本語で記述してください\n\n"
    "質問:{query_str}"
)


JP_QUESTION_GEN_QUERY = (
    "あなたはテストデータセット生成のための教師です。\n"
    "与えられた文書から、理解度を測るための{num_questions_per_chunk}個の質問を生成してください。\n"
    "## 質問は以下の要件を満たす必要があります\n"
    "- コンテキストに含まれる具体的な情報に基づいていること\n"
    "- コンテキストから直接回答できる質問であること\n"
    "- 明確で曖昧さのない表現を使用すること\n"
    "- コンテキストに記載されていない外部知識を必要としないこと\n"
    "- 以下のような多様な観点からの質問を含めること\n"
    "   1. 事実確認の質問（「～は何ですか？」「～はいつですか？」）\n"
    "   2. 理由・説明の質問（「なぜ～ですか？」「どのように～ですか？」）\n"
    "   3. 比較の質問（「～と～の違いは何ですか？」）\n"
    "   4. 評価の質問（「～の利点は何ですか？」）\n\n"
    "## 質問は必ず以下の形式で出力してください\n"
    "1. [質問文]\n"
    "2. [質問文]\n"
    "3. [質問文]\n"
)

JP_EVAL_TEMPLATE = PromptTemplate(
    "以下の情報がコンテキストによって支持されているかどうかを判断してください。\n"
    "回答は必ず YES または NO のいずれかで行ってください。\n"
    "たとえ大部分のコンテキストが無関係であっても、コンテキストのいずれかの部分が情報を支持していれば YES と答えてください。\n"
    "以下に例を示します。\n\n"
    "情報: アップルパイは一般的に二重のクラスト（上と下の生地）を持つ。\n"
    "コンテキスト: アップルパイは主要な具材がリンゴのフルーツパイです。\n"
    "アップルパイはしばしばホイップクリームやアイスクリーム（'apple pie à la mode'）、カスタード、チェダーチーズとともに提供されます。\n"
    "一般的に上と下の生地の両方があり、上のクラストは閉じているか格子模様です。\n"
    "回答: YES\n"
    "情報: アップルパイはまずい。\n"
    "コンテキスト: アップルパイは主要な具材がリンゴのフルーツパイです。\n"
    "アップルパイはしばしばホイップクリームやアイスクリーム（'apple pie à la mode'）、カスタード、チェダーチーズとともに提供されます。\n"
    "一般的に上と下の生地の両方があり、上のクラストは閉じているか格子模様です。\n"
    "回答: NO\n"
    "情報: {query_str}\n"
    "コンテキスト: {context_str}\n"
    "回答: "
)

JP_REFINE_TEMPLATE = PromptTemplate(
    "次の情報がコンテキストに含まれているか確認してください: {query_str}\n"
    "既に YES/NO の回答があります: {existing_answer}\n"
    "以下の追加コンテキストを使って、必要なら既存の回答を修正してください。\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "既存の回答が既に YES であれば、引き続き YES としてください。新しいコンテキストに情報が含まれていれば YES、そうでなければ NO と回答してください。\n"
)
