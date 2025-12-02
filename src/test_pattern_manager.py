import logging
import yaml
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pathlib import Path
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class TestPatternPaths(BaseModel):
    store_domain_patterns: str
    extractor_test_patterns: str

class TestPatternManager:
    def __init__(self, test_patterns_dir: str, config_manager: ConfigManager):
        self.test_patterns = {}
        self.test_patterns_dir = Path(test_patterns_dir)
        self.config_manager = config_manager
        self.paths = TestPatternPaths(
            store_domain_patterns=str(self.test_patterns_dir / "store_domain_patterns.yaml"),
            extractor_test_patterns=str(self.test_patterns_dir / "extractor_test_patterns.yaml"),
        )
        self._pattern_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        """YAMLファイルを読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.error(f"テストパターンファイルが見つかりません: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML解析エラー: {file_path}")
            raise
        except Exception as e:
            logger.error(f"テストパターンファイルの読み込み中にエラーが発生しました {file_path}: {e}")
            raise

    def load_all_test_patterns(self):
        self.test_patterns = {
            "extractor_test_patterns": self.load_yaml(self.paths.extractor_test_patterns),
            "store_domain_patterns": self.load_yaml(self.paths.store_domain_patterns),
        }
    
    def get_test_pattern(self, pattern_name: str, pattern_type: str, use_cache: bool = True) -> Dict[str, Any]:
        """特定のテストパターンを取得
        
        Args:
            pattern_name: パターン名（例: "store_glossary_basic"）
            pattern_type: パターンタイプ（例: "store_domain"）
            use_cache: キャッシュを使用するか
        """
        cache_key = f"{pattern_type}:{pattern_name}"
        if use_cache and cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
        
        if not self.test_patterns:
            self.load_all_test_patterns()
        
        pattern_file_key = f"{pattern_type}_patterns"
        patterns = self.test_patterns.get(pattern_file_key, {})
        pattern = patterns.get(pattern_name, {})
        
        if use_cache:
            self._pattern_cache[cache_key] = pattern
        return pattern

    # エクストラクタテストパターン関連
    def get_extractor_test_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """エクストラクタテストパターンを取得"""
        patterns = self.get_test_pattern("extractor_test_patterns")
        return patterns.get(pattern_name, {})
    
    def list_extractor_test_patterns(self) -> List[str]:
        """利用可能なエクストラクタテストパターン名のリストを取得"""
        patterns = self.get_test_pattern("extractor_test_patterns")
        return list(patterns.keys())
    
    # ストアドメイン知識テストパターン関連
    def get_store_domain_test_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """ストアドメイン知識テストパターンを取得"""
        patterns = self.get_test_pattern("store_domain_patterns")
        return patterns.get(pattern_name, {})
    
    def list_store_domain_test_patterns(self) -> List[str]:
        """利用可能なストアドメイン知識テストパターン名のリストを取得"""
        if not self.test_patterns:
            self.load_all_test_patterns()
        patterns = self.test_patterns.get("store_domain_patterns", {})
        return list(patterns.keys())
    
    def get_enabled_test_patterns(self, pattern_type: str) -> List[str]:
        """有効なテストパターン名のリストを取得"""
        if not self.test_patterns:
            self.load_all_test_patterns()
        
        pattern_file_key = f"{pattern_type}_patterns"
        patterns = self.test_patterns.get(pattern_file_key, {})
        
        enabled = []
        for name, config in patterns.items():
            if config.get("enabled", False):
                enabled.append(name)
        return enabled

    def get_store_domain_storage_config(self, pattern_name: str) -> Dict[str, Any]:
        """ストアドメイン知識テストパターンのストレージ設定を取得"""
        pattern = self.get_store_domain_test_pattern(pattern_name)
        return pattern.get("storage_config", {})

    def get_store_domain_indexing_patterns(self, pattern_name: str) -> List[Dict[str, Any]]:
        """ストアドメイン知識テストパターンのインデックスパターンリストを取得"""
        pattern = self.get_store_domain_test_pattern(pattern_name)
        return pattern.get("indexing_pattern", [])
    
    def get_store_domain_indexing_pattern_by_type(self, pattern_name: str, index_type: str) -> Dict[str, Any]:
        """特定のインデックスタイプのパターンを取得 (例: 'vector', 'summary')"""
        indexing_patterns = self.get_store_domain_indexing_patterns(pattern_name)
        for idx_pattern in indexing_patterns:
            if idx_pattern.get("type") == index_type:
                return idx_pattern
        return {}
    
    def get_store_domain_chunking_config_for_index(self, pattern_name: str, index_type: str) -> str:
        """特定のインデックスタイプのチャンキング設定名を取得"""
        idx_pattern = self.get_store_domain_indexing_pattern_by_type(pattern_name, index_type)
        return idx_pattern.get("chunking_config_model", "")
    
    def get_store_domain_extractor_pattern_for_index(self, pattern_name: str, index_type: str) -> str:
        """特定のインデックスタイプのエクストラクタパターン名を取得"""
        idx_pattern = self.get_store_domain_indexing_pattern_by_type(pattern_name, index_type)
        return idx_pattern.get("extractor_pattern", "")
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._pattern_cache.clear()
