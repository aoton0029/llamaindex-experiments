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


class PydanticOutputParserJp(PydanticOutputParser):
    """日本語対応のPydanticOutputParser"""
    
    def parse(self, output: str) -> Any:
        """日本語出力をパースしてPydanticモデルに変換"""
        # JSON文字列を抽出
        json_string = extract_json_str(output)
        
        try:
            json_dict = json.loads(json_string)
        except json.JSONDecodeError as e_json:
            try:
                import yaml
                # YAMLパーサーで再試行（末尾カンマなどに対応）
                json_dict = yaml.safe_load(json_string)
            except yaml.YAMLError as e_yaml:
                raise OutputParserException(
                    f"無効なJSON形式です。エラー: {e_json} {e_yaml}. "
                    f"取得したJSON文字列: {json_string}"
                )
            except NameError as exc:
                raise ImportError("PyYAMLをインストールしてください: pip install PyYAML") from exc
        
        try:
            # Pydanticモデルにパース
            return self.output_cls.parse_obj(json_dict)
        except Exception as e:
            raise OutputParserException(
                f"Pydanticモデルへの変換に失敗しました: {e}. "
                f"取得したデータ: {json_dict}"
            )

    def format(self, prompt_template: str) -> str:
        """プロンプトテンプレートにフォーマット指示を追加"""
        # Pydanticモデルのスキーマ情報を日本語で追加
        if hasattr(self.output_cls, 'schema'):
            schema = self.output_cls.schema()
            format_instruction = f"\n\n以下のJSON形式で出力してください:\n```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"
            return prompt_template + format_instruction
        return prompt_template