import json
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Type

from dataclasses_json import DataClassJsonMixin
from llama_index.core.output_parsers.base import (
    OutputParserException,
    StructuredOutput,
    ChainableOutputParser,
)
from llama_index.core.output_parsers.utils import _marshal_llm_to_json, extract_json_str
from llama_index.core.types import BaseOutputParser, Model
from llama_index.core.output_parsers.pydantic import PydanticOutputParser

def _escape_curly_braces(input_string: str) -> str:
    """波括弧をエスケープ"""
    return input_string.replace("{", "{{").replace("}", "}}")


@dataclass
class JapaneseAnswer(DataClassJsonMixin):
    """日本語対応の回答データクラス"""
    choice: int
    reason: str


class SelectionOutputParserJp(BaseOutputParser):
    """日本語対応のSelectionOutputParser"""
    
    REQUIRED_KEYS = frozenset(JapaneseAnswer.__annotations__)

    def _filter_dict(self, json_dict: dict) -> dict:
        """必要なキーを持つ辞書を再帰的に抽出"""
        output_dict = json_dict
        for key, val in json_dict.items():
            if key in self.REQUIRED_KEYS:
                continue
            elif isinstance(val, dict):
                output_dict = self._filter_dict(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        output_dict = self._filter_dict(item)
        return output_dict

    def _format_output(self, output: List[dict]) -> List[dict]:
        """出力を整形"""
        output_json = []
        for json_dict in output:
            valid = True
            for key in self.REQUIRED_KEYS:
                if key not in json_dict:
                    valid = False
                    break

            if not valid:
                json_dict = self._filter_dict(json_dict)

            output_json.append(json_dict)

        return output_json

    def parse(self, output: str) -> Any:
        """日本語出力をパース"""
        # JSON文字列を抽出
        json_string = _marshal_llm_to_json(output)
        
        try:
            json_obj = json.loads(json_string)
        except json.JSONDecodeError as e_json:
            try:
                import yaml
                # YAMLパーサーで再試行（末尾カンマなどに対応）
                json_obj = yaml.safe_load(json_string)
            except yaml.YAMLError as e_yaml:
                raise OutputParserException(
                    f"無効なJSON形式です。エラー: {e_json} {e_yaml}. "
                    f"取得したJSON文字列: {json_string}"
                )
            except NameError as exc:
                raise ImportError("PyYAMLをインストールしてください: pip install PyYAML") from exc

        # 辞書の場合はリストに変換
        if isinstance(json_obj, dict):
            json_obj = [json_obj]

        if not isinstance(json_obj, list):
            raise ValueError(f"出力をJSONに変換できませんでした: {output!r}")

        # 出力を整形
        json_output = self._format_output(json_obj)
        
        # Answerオブジェクトに変換
        answers = [JapaneseAnswer.from_dict(json_dict) for json_dict in json_output]
        
        return StructuredOutput(raw_output=output, parsed_output=answers)

    def format(self, prompt_template: str) -> str:
        """プロンプトテンプレートにフォーマット指示を追加"""
        return prompt_template #+ "\n\n" + _escape_curly_braces(JP_FORMAT_STR)


# 日本語用のフォーマットテンプレート
PYDANTIC_FORMAT_TMPL_JP = """
以下のJSONスキーマに従ってください:
{schema}

有効なJSONオブジェクトを出力してください。スキーマは繰り返さないでください。
"""


class PydanticOutputParserJp(ChainableOutputParser, Generic[Model]):
    """
    日本語対応のPydantic Output Parser.

    Args:
        output_cls (BaseModel): Pydantic出力クラス
        excluded_schema_keys_from_format: フォーマットから除外するスキーマキー
        pydantic_format_tmpl: フォーマットテンプレート（日本語）

    """

    def __init__(
        self,
        output_cls: Type[Model],
        excluded_schema_keys_from_format: Optional[List] = None,
        pydantic_format_tmpl: str = PYDANTIC_FORMAT_TMPL_JP,
    ) -> None:
        """初期化"""
        self._output_cls = output_cls
        self._excluded_schema_keys_from_format = excluded_schema_keys_from_format or []
        self._pydantic_format_tmpl = pydantic_format_tmpl

    @property
    def output_cls(self) -> Type[Model]:
        """出力クラスを取得"""
        return self._output_cls

    @property
    def format_string(self) -> str:
        """フォーマット文字列を取得"""
        return self.get_format_string(escape_json=True)

    def get_format_string(self, escape_json: bool = True) -> str:
        """
        フォーマット文字列を生成
        
        Args:
            escape_json: JSONの波括弧をエスケープするか
            
        Returns:
            フォーマット文字列
        """
        schema_dict = self._output_cls.model_json_schema()
        for key in self._excluded_schema_keys_from_format:
            del schema_dict[key]

        schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2)
        output_str = self._pydantic_format_tmpl.format(schema=schema_str)
        if escape_json:
            return output_str.replace("{", "{{").replace("}", "}}")
        else:
            return output_str

    def parse(self, text: str) -> Any:
        """
        パース、検証、プログラム的なエラー修正
        
        Args:
            text: LLMからの出力テキスト
            
        Returns:
            パースされたPydanticモデルインスタンス
        """
        json_str = extract_json_str(text)
        try:
            return self._output_cls.model_validate_json(json_str)
        except Exception as e:
            # YAMLパーサーで再試行（日本語LLMの出力に対応）
            try:
                import yaml
                json_obj = yaml.safe_load(json_str)
                return self._output_cls.model_validate(json_obj)
            except yaml.YAMLError:
                raise OutputParserException(
                    f"JSONのパースに失敗しました。エラー: {e}\n"
                    f"取得したJSON文字列: {json_str}"
                )
            except NameError as exc:
                raise ImportError("PyYAMLをインストールしてください: pip install PyYAML") from exc

    def format(self, query: str) -> str:
        return query + "\n\n" + self.get_format_string(escape_json=True)