import logging
from typing import List, Optional, Dict, Any
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from src.adapters.llamaindex.factories import (
    QueryEngineFactory,
    ResponseSynthesizerFactory,
)
from src.adapters.llamaindex.settings import DomainLLMSettings
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


# システムプロンプト
SYSTEM_PROMPT = """あなたは営業支援AIアシスタントです。
営業と取引先の会話記録から、取引先のニーズや要望を正確に抽出して回答してください。

回答時の注意点：
- 会話記録に基づいた事実のみを回答
- 推測や憶測は避ける
- 具体的な会社名や担当者名も含めて回答
- 複数の会話に関連情報がある場合は統合して回答
"""

# クエリプロンプトテンプレート
QA_PROMPT_TEMPLATE = """以下の会話記録から、質問に関する情報を抽出してください。

会話記録：
{context_str}

質問：{query_str}

回答形式：
- 取引先名：
- 担当者名：
- ニーズ・要望：
- 決定事項：
- 関連する会話の日時：

回答："""


class ConversationQueryEngine:
    """会話情報用QueryEngineファクトリー"""

    @staticmethod
    def create(
        retriever: BaseRetriever,
        response_mode: str = "compact",
        use_custom_prompt: bool = True,
    ):
        """
        QueryEngineを作成

        Args:
            retriever: カスタムRetriever
            response_mode: 回答生成戦略（compact/refine/tree_summarize）
            use_custom_prompt: カスタムプロンプトを使用するか

        Returns:
            RetrieverQueryEngine
        """
        # ResponseSynthesizerの作成
        response_synthesizer = ResponseSynthesizerFactory.get(
            llm=DomainLLMSettings.SYNTHESIZER_RESPONSE,
            response_mode=response_mode,
        )

        QueryEngineFactory.create()
        # QueryEngineの作成
        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=response_synthesizer
        )

        # カスタムプロンプトの設定
        if use_custom_prompt:
            qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt}
            )

        logger.info(f"QueryEngine作成完了: mode={response_mode}, custom_prompt={use_custom_prompt}")

        return query_engine
