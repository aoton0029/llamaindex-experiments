import logging
import yaml
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pathlib import Path
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

class TestPatternPaths(BaseModel):
    domain_knowledge_test_patterns: str
    indexing_test_patterns: str

class TestPatternManager:
    def __init__(self, test_patterns_dir: str, config_manager: ConfigManager):
        self.test_patterns = {}
        self.test_patterns_dir = Path(test_patterns_dir)
        self.config_manager = config_manager
        self.paths = TestPatternPaths(
            domain_knowledge_test_patterns=str(self.test_patterns_dir / "domain_knowledge_test_patterns.yaml"),
            indexing_test_patterns=str(self.test_patterns_dir / "indexing_test_patterns.yaml"),
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
            "indexing_test_patterns": self.load_yaml(self.paths.indexing_test_patterns),
            "domain_knowledge_test_patterns": self.load_yaml(self.paths.domain_knowledge_test_patterns),
        }
    
    def get_test_pattern(self, test_pattern_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """テストパターンファイルから読込
        
        Args:
            test_pattern_name: テストパターンファイル
            use_cache: キャッシュを使用するか
        """
        if use_cache and test_pattern_name in self._pattern_cache:
            return self._pattern_cache[test_pattern_name]
        
        if not self.test_patterns:
            self.load_all_test_patterns()
        
        patterns = self.test_patterns.get(test_pattern_name, {})
        
        if use_cache:
            self._pattern_cache[test_pattern_name] = patterns
        
        return patterns

    # インデクシングテストパターン関連
    
    
    # ドメイン知識テストパターン関連
    def get_domain_knowledge_test_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """ドメイン知識テストパターンを取得"""
        patterns = self.get_test_pattern("domain_knowledge_test_patterns")
        return patterns.get(pattern_name, {})
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._pattern_cache.clear()