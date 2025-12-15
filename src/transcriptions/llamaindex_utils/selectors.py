import logging
from typing import Any, List, Dict, Optional
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.node import BaseNodePostprocessor
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.selectors.prompts import SingleSelectPrompt, MultiSelectPrompt

from .output_parser_factory import SelectionOutputParserJp


logger = logging.getLogger(__name__)

JP_SINGLE_SELECT_PROMPT_TMPL = """
あなたは高度な情報検索システムのクエリルーターです。
質問を分析し、最も適切な情報源を1つだけ選択し、JSON形式で回答する専門家です。

# 指示
1. 質問を分析し、最も適切な情報源を「一つだけ」選べ
2. 質問の主要なトピックと最も関連性が高い選択肢を選べ
3. 複数該当する場合は、より具体的で詳細な情報源を優先せよ
4. どれも該当しない場合は、最も一般的な選択肢を選べ
5. 選択した理由を簡潔に日本語で説明せよ
6. 必ず以下のJSON形式で回答せよ

# 出力形式
{
    "choice": <選択した番号（整数）>,
    "reason": "<選択理由を日本語で簡潔に説明>"
}

# 注意事項
- JSON形式以外の出力は一切不要
- choice: 必ず整数型の番号
- reason: 選択理由を1～2文で簡潔に記述
- 余計な説明、記号、改行を含めない
- 正しいJSON文法に従うこと（ダブルクォート使用、末尾カンマなし）

# 例
質問：Pythonのリスト操作について教えて
選択肢：
1. Python基礎文法ガイド - Pythonの基本的な文法と構文を解説
2. データ構造とアルゴリズム - リスト、辞書、セットなどのデータ構造の詳細
3. Web開発フレームワーク - Django、Flaskなどのフレームワーク解説

出力:
{
    "choice": 2,
    "reason": "リスト操作はデータ構造の一部であり、選択肢2が最も具体的で詳細な情報を提供するため"
}


質問：{query_str}
選択肢：{context_list}

出力:
"""

JP_MULTI_SELECT_PROMPT_TMPL = """
あなたは情報検索システムのマルチクエリルーターです。
質問を分析し、適切な情報源を複数選択し、JSON形式で回答する専門家です。

# 指示
1. 質問を分析し、適切な情報源を**複数（0個、1個または複数個）**選択せよ
2. 関連性が高い順に選択せよ（最も関連性が高いものを最初に）
3. 関連性が低いまたは無関係な選択肢は含めない
4. 選択可能な番号範囲：1～{num_choices}
5. 最大{max_outputs}個まで選択可能（超えないこと）
6. 各選択について理由を簡潔に日本語で説明せよ
7. 必ず以下のJSON形式で回答せよ

# 出力形式
[
    {
        "choice": <選択した番号1（整数）>,
        "reason": "<選択理由1を日本語で簡潔に説明>"
    },
    {
        "choice": <選択した番号2（整数）>,
        "reason": "<選択理由2を日本語で簡潔に説明>"
    }
]

# 注意事項
- JSON形式以外の出力は一切不要
- choice: 必ず整数型の番号
- reason: 選択理由を1～2文で簡潔に記述
- 関連性の高い順に並べる
- 余計な説明、記号、改行を含めない
- 正しいJSON文法に従うこと（ダブルクォート使用、最後の要素に末尾カンマなし）
- 該当なしの場合は空配列 [] を返す

# 例
質問：Pythonでのデータ分析について教えて
選択肢：
1. Python基礎文法ガイド - Pythonの基本的な文法と構文を解説
2. データ分析ライブラリ - pandas、numpy、matplotlibの使い方
3. 機械学習入門 - scikit-learnを使った機械学習の基礎
4. Web開発フレームワーク - Django、Flaskなどのフレームワーク解説
出力:
[
    {
        "choice": 2,
        "reason": "データ分析に直接関連するライブラリの詳細情報を提供するため最も適切"
    },
    {
        "choice": 3,
        "reason": "データ分析の発展的な内容として機械学習は関連性が高い"
    },
    {
        "choice": 1,
        "reason": "Python基礎は補助的な情報として有用"
    }
]


質問：{query_str}
選択肢：{context_list}

出力:
"""

class LLMSingleSelectorJp(LLMSingleSelector):
    """日本語対応LLM Single Selector"""
    
    @classmethod
    def from_defaults(cls, 
        llm: Optional[LLM] = None,
    ) -> "LLMSingleSelectorJp":
        """日本語デフォルト設定でインスタンスを作成"""
        prompt_template_str = JP_SINGLE_SELECT_PROMPT_TMPL
        output_parser = SelectionOutputParserJp()
        
        prompt = SingleSelectPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SINGLE_SELECT,
        )
        
        logger.info("LLMSingleSelectorJp（日本語対応シングルセレクター）を作成")
        return cls(llm=llm, prompt=prompt)


class LLMMultiSelectorJp(LLMMultiSelector):
    """日本語対応LLM Multi Selector"""
    
    @classmethod
    def from_defaults(cls,
        llm: Optional[LLM] = None,
        max_outputs: Optional[int] = None) -> "LLMMultiSelectorJp":
        """日本語デフォルト設定でインスタンスを作成"""
        prompt_template_str = JP_MULTI_SELECT_PROMPT_TMPL
        output_parser = SelectionOutputParserJp()
        prompt_template_str = output_parser.format(prompt_template_str)
        
        prompt = MultiSelectPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
           prompt_type=PromptType.MULTI_SELECT,
        )
        
        logger.info("LLMMultiSelectorJp（日本語対応マルチセレクター）を作成")
        return cls(llm=llm, prompt=prompt, max_outputs=max_outputs)