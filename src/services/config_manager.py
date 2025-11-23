import logging
import yaml
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigPaths(BaseModel):
    chunking: str
    embedding: str
    llm: str
    tokenizer: str
    evaluation: str
    schemas: str
    extractor: str
    templates: str
    chunk_test_patterns: str
    evaluation_test_patterns: str
    extractor_test_patterns: str


class ConfigManager:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.paths = ConfigPaths(
            chunking=str(self.config_dir / "chunking_configs.yaml"),
            embedding=str(self.config_dir / "embedding_configs.yaml"),
            llm=str(self.config_dir / "llm_configs.yaml"),
            tokenizer=str(self.config_dir / "tokenizer_configs.yaml"),
            evaluation=str(self.config_dir / "evaluation_configs.yaml"),
            schemas=str(self.config_dir / "schema_config.yaml"),
            extractor=str(self.config_dir / "extractor_configs.yaml"),
            templates=str(self.config_dir / "templates.yaml"),
            evaluation_test_patterns=str(Path(config_dir).parent / "tests" / "evaluation_test_patterns.yaml"),
            extractor_test_patterns=str(Path(config_dir).parent / "tests" / "extractor_test_patterns.yaml"),
        )
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        

    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"設定ファイルが見つかりません: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML解析エラー: {file_path}")
            raise
        except Exception as e:
            logger.error(f"設定ファイルの読み込み中にエラーが発生しました {file_path}: {e}")
            raise

    def load_all_configs(self):
        self.configs = {
            "chunking": self.load_yaml(self.paths.chunking),
            "embedding": self.load_yaml(self.paths.embedding),
            "llm": self.load_yaml(self.paths.llm),
            "tokenizer": self.load_yaml(self.paths.tokenizer),
            "evaluation": self.load_yaml(self.paths.evaluation),
            "schema": self.load_yaml(self.paths.schemas),
            "extractor": self.load_yaml(self.paths.extractor),
            "templates": self.load_yaml(self.paths.templates),
            "evaluation_test_patterns": self.load_yaml(self.paths.evaluation_test_patterns),
            "extractor_test_patterns": self.load_yaml(self.paths.extractor_test_patterns),
        }


    def get_config(self, config_type: str, use_cache: bool = True) -> Dict[str, Any]:
        """項目名からconfigファイルの内容を取得"""
        if use_cache and config_type in self._config_cache:
            return self._config_cache[config_type]
        
        if config_type not in self.configs:
            self.load_all_configs()
        
        config = self.configs.get(config_type, {})
        if use_cache:
            self._config_cache[config_type] = config
        return config

    # 各コンフィグファイルから特定のコンフィグ取得
    def get_tokenizer_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("tokenizer")
        cfg_models = config.get("tokenizer_config_models", {})
        return cfg_models.get(type_name, {})

    def get_chunking_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("chunking")
        cfg_models = config.get("chunker_config_models", {})
        return cfg_models.get(type_name, {})

    def get_embedding_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("embedding")
        cfg_models = config.get("embedding_config_models", {})
        return cfg_models.get(type_name, {})

    def get_llm_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("llm")
        cfg_models = config.get("llm_config_models", {})
        return cfg_models.get(type_name, {})

    def get_evaluation_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("evaluation")
        cfg_models = config.get("evaluation_config_models", {})
        return cfg_models.get(type_name, {})
    
    def get_schema_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("schema")
        cfg_models = config.get("schema_config_models", {})
        return cfg_models.get(type_name, {})
    
    def get_indexing_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("indexing")
        cfg_models = config.get("indexing_config_models", {})
        return cfg_models.get(type_name, {})
    
    # テストパターン関連
    def get_evaluation_test_patterns(self) -> Dict[str, Any]:
        """評価テストパターンを取得"""
        return self.get_config("evaluation_test_patterns")
    
    def get_test_pattern(self, pattern_name: str, pattern_type: str = "evaluation") -> Dict[str, Any]:
        """
        指定されたテストパターンを取得
        
        Args:
            pattern_name: パターン名
            pattern_type: パターンタイプ ("chunk", "evaluation", "extractor")
            
        Returns:
            テストパターンの設定
        """
        if pattern_type == "extractor":
            patterns = self.get_extractor_test_patterns()
        else:
            patterns = self.get_evaluation_test_patterns()
        
        return patterns.get(pattern_name, {})
    
    def get_tokenizer_config_from_pattern(self, pattern_name: str, pattern_type: str = "evaluation") -> Dict[str, Any]:
        """テストパターンからトークナイザー設定を取得"""
        pattern = self.get_test_pattern(pattern_name, pattern_type)
        tokenizer_type = pattern.get("tokenizer_config_model", {}).get("type")
        if tokenizer_type:
            return self.get_tokenizer_config(tokenizer_type)
        return {}
    
    def get_llm_config_from_pattern(self, pattern_name: str, pattern_type: str = "evaluation") -> Dict[str, Any]:
        """テストパターンからLLM設定を取得"""
        pattern = self.get_test_pattern(pattern_name, pattern_type)
        llm_type = pattern.get("llm_config_model", {}).get("type")
        if llm_type:
            return self.get_llm_config(llm_type)
        return {}
    
    def get_embedding_config_from_pattern(self, pattern_name: str, pattern_type: str = "evaluation") -> Dict[str, Any]:
        """テストパターンから埋め込み設定を取得"""
        pattern = self.get_test_pattern(pattern_name, pattern_type)
        embedding_type = pattern.get("embedding_config_model", {}).get("type")
        if embedding_type:
            return self.get_embedding_config(embedding_type)
        return {}
    
    def get_chunking_config_from_pattern(self, pattern_name: str, pattern_type: str = "chunk") -> Dict[str, Any]:
        """テストパターンからチャンキング設定を取得"""
        pattern = self.get_test_pattern(pattern_name, pattern_type)
        chunking_type = pattern.get("chunking_config_model", {}).get("type")
        if chunking_type:
            return self.get_chunking_config(chunking_type)
        return {}
    
    def get_indexing_strategy_from_pattern(self, pattern_name: str, pattern_type: str = "evaluation") -> List[Dict[str, Any]]:
        """テストパターンからインデックス戦略を取得"""
        pattern = self.get_test_pattern(pattern_name, pattern_type)
        return pattern.get("indexing_strategy", [])
    
    def get_evaluation_strategy_from_pattern(self, pattern_name: str, pattern_type: str = "evaluation") -> Dict[str, Any]:
        """テストパターンから評価戦略を取得"""
        pattern = self.get_test_pattern(pattern_name, pattern_type)
        return pattern.get("evaluation_strategy", {})
    
    # エクストラクタ関連
    def get_extractor_config(self, type_name: str) -> Dict[str, Any]:
        """エクストラクタ設定を取得"""
        config = self.get_config("extractor")
        cfg_models = config.get("extractor_config_models", {})
        return cfg_models.get(type_name, {})
    
    def get_extractor_pattern_config(self, pattern_name: str) -> Dict[str, Any]:
        """エクストラクタパターン設定を取得"""
        config = self.get_config("extractor")
        patterns = config.get("extractor_patterns", {})
        return patterns.get(pattern_name, {})
    
    def get_extractor_test_patterns(self) -> Dict[str, Any]:
        """エクストラクタテストパターンを取得"""
        return self.get_config("extractor_test_patterns")
    
    def get_extractor_pattern_from_test(self, pattern_name: str) -> Dict[str, Any]:
        """テストパターンからエクストラクタパターンを取得"""
        pattern = self.get_test_pattern(pattern_name, "extractor")
        extractor_pattern_type = pattern.get("extractor_pattern", {}).get("type")
        if extractor_pattern_type:
            return self.get_extractor_pattern_config(extractor_pattern_type)
        return {}
    
    # テンプレート関連
    def get_template_prompts(self) -> Dict[str, Any]:
        """テンプレートプロンプト設定を取得"""
        config = self.get_config("templates")
        return config.get("template_prompts", {})
    
    def get_template_prompt_category(self, category: str) -> Dict[str, Any]:
        """特定カテゴリのテンプレートプロンプトを取得"""
        templates = self.get_template_prompts()
        return templates.get(category, {})
    
    def get_template_prompt(self, category: str, template_name: str) -> str:
        """特定のテンプレートプロンプトを取得"""
        category_templates = self.get_template_prompt_category(category)
        return category_templates.get(template_name, "")
    
