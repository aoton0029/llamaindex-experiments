import re
import logging
from typing import List, Union, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod

from llama_index.core import settings
from llama_index.core.schema import Document, BaseNode


logger = logging.getLogger(__name__)


class BasePreProcessor(ABC):
    """前処理の基底クラス"""
    
    @abstractmethod
    def process(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """ドキュメントを前処理"""
        pass


class MetadataPreProcessor(BasePreProcessor):
    """メタデータの整形を行う前処理"""
    
    def __init__(self, 
                 normalize_keys: bool = True,
                 remove_empty: bool = True,
                 key_mapping: Optional[Dict[str, str]] = None,
                 default_metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            normalize_keys: キー名を正規化（小文字化、スペース除去）
            remove_empty: 空のメタデータを削除
            key_mapping: キー名の変換マッピング
            default_metadata: デフォルトで追加するメタデータ
        """
        self.normalize_keys = normalize_keys
        self.remove_empty = remove_empty
        self.key_mapping = key_mapping or {}
        self.default_metadata = default_metadata or {}
    
    def process(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """メタデータを整形"""
        flat_docs = self._flatten_documents(documents)
        processed_docs = []
        
        for doc in flat_docs:
            # デフォルトメタデータを追加
            new_metadata = self.default_metadata.copy()
            
            for key, value in doc.metadata.items():
                # キー名を正規化
                new_key = self._normalize_key(key) if self.normalize_keys else key
                
                # キーマッピングを適用
                new_key = self.key_mapping.get(new_key, new_key)
                
                # 空の値をスキップ
                if self.remove_empty and not value:
                    continue
                
                new_metadata[new_key] = value
            
            doc.metadata = new_metadata
            processed_docs.append(doc)
        
        logger.info(f"{len(processed_docs)}件のドキュメントのメタデータを整形")
        return processed_docs
    
    def _normalize_key(self, key: str) -> str:
        """キー名を正規化"""
        return key.lower().strip().replace(" ", "_")
    
    def _flatten_documents(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """ネストされたドキュメントリストをフラット化"""
        if not documents:
            return []
        
        if isinstance(documents[0], list):
            return [doc for sublist in documents for doc in sublist]
        return documents


class TOCPreProcessor(BasePreProcessor):
    """目次（Table of Contents）の整形を行う前処理"""
    
    def __init__(self,
                 remove_toc: bool = False,
                 toc_patterns: Optional[List[str]] = None,
                 extract_toc_as_metadata: bool = True):
        """
        Args:
            remove_toc: 目次を削除するか
            toc_patterns: 目次を検出する正規表現パターン
            extract_toc_as_metadata: 目次をメタデータとして抽出
        """
        self.remove_toc = remove_toc
        self.toc_patterns = toc_patterns or [
            # 日本語の目次パターン（複数行対応）
            r'(?:目\s*次|#{1,6}\s*目\s*次)\s*\n(?:.*?\n)*?(?=\n{2,}|第?\d+[章節]|\d+-\d+|[A-Z]\.|\Z)',
            # 番号付きリスト形式（1-1, 1-2など）
            r'(?:目\s*次|#{1,6}\s*目\s*次)[\s\S]*?(?:\d+-\d+\s+.+?\s+\.+\s+\d+\s*\n)+',
            r"目次\n.*?(?=\n\n|\Z)",
            r"Table of Contents\n.*?(?=\n\n|\Z)",
            r"Contents\n.*?(?=\n\n|\Z)",
            r"^\d+\..*?(?=\n\d+\.|\Z)",  # 番号付きリスト
        ]
        self.extract_toc_as_metadata = extract_toc_as_metadata
    
    def process(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """目次を処理"""
        flat_docs = self._flatten_documents(documents)
        processed_docs = []
        
        for doc in flat_docs:
            text = doc.text
            toc_content = None
            
            # 目次を検出
            for pattern in self.toc_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    toc_content = match.group(0)
                    
                    if self.remove_toc:
                        text = text[:match.start()] + text[match.end():]
                    
                    if self.extract_toc_as_metadata and toc_content:
                        doc.metadata["table_of_contents"] = toc_content.strip()
                    
                    break
            
            doc.text = text.strip()
            processed_docs.append(doc)
        
        logger.info(f"{len(processed_docs)}件のドキュメントの目次を処理")
        return processed_docs
    
    def _flatten_documents(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """ネストされたドキュメントリストをフラット化"""
        if not documents:
            return []
        
        if isinstance(documents[0], list):
            return [doc for sublist in documents for doc in sublist]
        return documents


class TextCleanerPreProcessor(BasePreProcessor):
    """不要な文字列を削除する前処理"""
    
    def __init__(self,
                 remove_patterns: Optional[List[str]] = None,
                 remove_extra_whitespace: bool = True,
                 remove_page_numbers: bool = True,
                 normalize_unicode: bool = True):
        """
        Args:
            remove_patterns: 削除する正規表現パターンのリスト
            remove_extra_whitespace: 余分な空白を削除
            remove_page_numbers: ページ番号を削除
            normalize_unicode: Unicode文字を正規化
        """
        self.remove_patterns = remove_patterns or []
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_page_numbers = remove_page_numbers
        self.normalize_unicode = normalize_unicode
        
        # デフォルトパターンを追加
        if self.remove_page_numbers:
            self.remove_patterns.extend([
                r'(?:ページ|Page)\s*\d+',
                r'^\s*\d+\s*$',  # 単独の数字行
                r'\[\d+\]',  # [1], [2]などの参照番号
            ])
    
    def process(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """テキストをクリーニング"""
        flat_docs = self._flatten_documents(documents)
        processed_docs = []
        
        for doc in flat_docs:
            text = doc.text
            
            # Unicode正規化
            if self.normalize_unicode:
                import unicodedata
                text = unicodedata.normalize('NFKC', text)
            
            # パターンマッチングで削除
            for pattern in self.remove_patterns:
                text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
            
            # 余分な空白を削除
            if self.remove_extra_whitespace:
                text = re.sub(r'\n{3,}', '\n\n', text)  # 3つ以上の改行を2つに
                text = re.sub(r' {2,}', ' ', text)  # 2つ以上のスペースを1つに
                text = re.sub(r'\t+', ' ', text)  # タブをスペースに
            
            doc.text = text.strip()
            processed_docs.append(doc)
        
        logger.info(f"{len(processed_docs)}件のドキュメントをクリーニング")
        return processed_docs
    
    def _flatten_documents(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """ネストされたドキュメントリストをフラット化"""
        if not documents:
            return []
        
        if isinstance(documents[0], list):
            return [doc for sublist in documents for doc in sublist]
        return documents


class CompositePreProcessor(BasePreProcessor):
    """複数の前処理を組み合わせる"""
    
    def __init__(self, preprocessors: List[BasePreProcessor]):
        """
        Args:
            preprocessors: 実行する前処理のリスト（実行順）
        """
        self.preprocessors = preprocessors
    
    def process(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """複数の前処理を順次実行"""
        result = documents
        
        for preprocessor in self.preprocessors:
            result = preprocessor.process(result)
        
        return result

    
    def process(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """スキーマに基づいてメタデータを整形"""
        flat_docs = self._flatten_documents(documents)
        processed_docs = []
        
        for doc in flat_docs:
            new_metadata = {}
            
            # スキーマに定義されたフィールドを処理
            for field_name, field_info in self.schema_fields.items():
                if field_name in doc.metadata:
                    # 既存の値を検証・変換
                    if self.validate_types:
                        value = self._validate_and_convert_value(
                            doc.metadata[field_name], 
                            field_info
                        )
                    else:
                        value = doc.metadata[field_name]
                    
                    new_metadata[field_name] = value
                
                elif self.fill_missing:
                    # 欠損フィールドをデフォルト値で埋める
                    new_metadata[field_name] = field_info["default"]
            
            # 厳格モードでない場合、スキーマにないフィールドも保持
            if not self.strict_mode:
                for key, value in doc.metadata.items():
                    if key not in new_metadata:
                        new_metadata[key] = value
            
            doc.metadata = new_metadata
            processed_docs.append(doc)
        
        logger.info(f"{len(processed_docs)}件のドキュメントをスキーマに基づいて整形")
        return processed_docs
    
    def _flatten_documents(self, documents: Union[List[Document], List[List[Document]]]) -> List[Document]:
        """ネストされたドキュメントリストをフラット化"""
        if not documents:
            return []
        
        if isinstance(documents[0], list):
            return [doc for sublist in documents for doc in sublist]
        return documents


class PreProcessorFactory:
    """前処理ファクトリー"""
    
    @staticmethod
    def create(processor_type: str, **kwargs) -> BasePreProcessor:
        """
        前処理を作成
        
        Args:
            processor_type: 前処理のタイプ
                - "metadata": メタデータ整形
                - "schema_based": スキーマベースのメタデータ整形
                - "metadata_filter": メタデータフィルタリング
                - "toc": 目次処理
                - "cleaner": テキストクリーニング
                - "composite": 複合処理
        """
        if processor_type == "metadata":
            return MetadataPreProcessor(**kwargs)
        elif processor_type == "toc":
            return TOCPreProcessor(**kwargs)
        elif processor_type == "cleaner":
            return TextCleanerPreProcessor(**kwargs)
        elif processor_type == "composite":
            return CompositePreProcessor(**kwargs)
        else:
            raise ValueError(f"未知の前処理タイプ: {processor_type}")
    
