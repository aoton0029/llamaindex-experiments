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
    extractor: str
    templates: str
    prompt_helper: str

class ConfigManager:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.paths = ConfigPaths(
            chunking=str(self.config_dir / "chunking_configs.yaml"),
            embedding=str(self.config_dir / "embedding_configs.yaml"),
            llm=str(self.config_dir / "llm_configs.yaml"),
            tokenizer=str(self.config_dir / "tokenizer_configs.yaml"),
            extractor=str(self.config_dir / "extractor_configs.yaml"),
            templates=str(self.config_dir / "templates.yaml"),
            prompt_helper=str(self.config_dir / "prompt_helper_configs.yaml"),
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
            "extractor": self.load_yaml(self.paths.extractor),
            "templates": self.load_yaml(self.paths.templates),
            "prompt_helper": self.load_yaml(self.paths.prompt_helper),
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

    def get_llm_domain_config(self) -> Dict[str, Any]:
        config = self.get_config("llm")
        cfg_models = config.get("llm_domain_kwargs_models", {})
        return cfg_models

    def get_prompt_helper_config(self, type_name: str) -> Dict[str, Any]:
        """プロンプトヘルパー設定を取得"""
        config = self.get_config("prompt_helper")
        cfg_models = config.get("prompt_helper_config_models", {})
        return cfg_models.get(type_name, {})

    def get_indexing_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("indexing")
        cfg_models = config.get("indexing_config_models", {})
        return cfg_models.get(type_name, {})
    
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


