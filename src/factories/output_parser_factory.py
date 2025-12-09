import json
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Type

from dataclasses_json import DataClassJsonMixin
from llama_index.core.output_parsers import SelectionOutputParser
from llama_index.core.output_parsers.base import (
    OutputParserException,
    StructuredOutput,
)
from llama_index.core.output_parsers.utils import extract_json_str
from llama_index.core.types import BaseOutputParser, Model
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from .settings_template_prompts import TemplatePromptSettings

def _escape_curly_braces(input_string: str) -> str:
    """波括弧をエスケープ"""
    return input_string.replace("{", "{{").replace("}", "}}")


@dataclass
class JapaneseAnswer(DataClassJsonMixin):
    """日本語対応の回答データクラス"""
    choice: int
    reason: str


class SelectionOutputParserJp(SelectionOutputParser):
    """日本語対応のSelectionOutputParser"""
    
    REQUIRED_KEYS = frozenset(JapaneseAnswer.__annotations__)
    
    def parse(self, output: str) -> Any:
        """日本語出力をパースして回答リストに変換"""
        json_string = extract_json_str(output)
        
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
            raise ValueError(f"JSON配列への変換に失敗しました: {output!r}")
        
        # フォーマットを整形
        json_output = self._format_output(json_obj)
        answers = [JapaneseAnswer.from_dict(json_dict) for json_dict in json_output]
        return StructuredOutput(raw_output=output, parsed_output=answers)
    
    def format(self, prompt_template: str) -> str:
        """プロンプトテンプレートに日本語フォーマット指示を追加"""
        return prompt_template #+ "\n\n" + _escape_curly_braces(TemplatePromptSettings.SELECTION_FORMAT_INSTRUCTION_JP)

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