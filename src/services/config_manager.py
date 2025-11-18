import logging
import yaml
from typing import Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigPaths(BaseModel):
    chunking: str
    embedding: str
    llm: str
    tokenizer: str
    evaluation: str
    test_patterns: str
    schemas: str


class ConfigManager:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.paths = ConfigPaths(
            chunking=str(self.config_dir / "chunking_config.yaml"),
            embedding=str(self.config_dir / "embedding_config.yaml"),
            llm=str(self.config_dir / "llm_config.yaml"),
            tokenizer=str(self.config_dir / "tokenizer_config.yaml"),
            evaluation=str(self.config_dir / "evaluation_config.yaml"),
            schemas=str(self.config_dir / "schema_config.yaml"),
            test_patterns=str(self.config_dir / "test_patterns.yaml"),
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
            "test_patterns": self.load_yaml(self.paths.test_patterns),
            "schema": self.load_yaml(self.paths.schemas),
        }

    def get_config(self, config_type: str, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and config_type in self._config_cache:
            return self._config_cache[config_type]
        
        if config_type not in self.configs:
            self.load_all_configs()
        
        config = self.configs.get(config_type, {})
        if use_cache:
            self._config_cache[config_type] = config
        return config

    def get_test_patterns(self) -> Dict[str, Any]:
        return self.get_config("test_patterns")

    def get_experiment_pattern(self, pattern_name: str) -> Dict[str, Any]:
        config = self.get_config("test_patterns")
        return config.get(pattern_name, {})
    
    def get_tokenizer_config_pattern(self, pattern_name:str) -> Dict[str, Any]:
        pat = self.get_experiment_pattern(pattern_name)
        tokenizer_type = pat["tokenizer_config"]["type"]
        return self.get_tokenizer_config(tokenizer_type)
    
    def get_chunking_config_pattern(self, pattern_name:str) -> Dict[str, Any]:
        pat = self.get_experiment_pattern(pattern_name)
        chunking_type = pat["chunker_config"]["type"]
        return self.get_chunking_config(chunking_type)
    
    def get_embedding_config_pattern(self, pattern_name:str) -> Dict[str, Any]:
        pat = self.get_experiment_pattern(pattern_name)
        embedding_type = pat["embedding_config"]["type"]
        return self.get_embedding_config(embedding_type)
    
    def get_llm_config_pattern(self, pattern_name:str) -> Dict[str, Any]:
        pat = self.get_experiment_pattern(pattern_name)
        llm_type = pat["llm_config"]["type"]
        return self.get_llm_config(llm_type)
    
    def get_indexing_config_pattern(self, pattern_name: str) -> Dict[str, Any]:
        pat = self.get_experiment_pattern(pattern_name)
        return 
        
    def get_schema_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("schema")
        return config.get(type_name, {})
    

    # 各コンフィグファイルから特定のコンフィグ取得
    def get_tokenizer_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("tokenizer")
        types = config.get("tokenizer_types", {})
        return types.get(type_name, {})

    def get_chunking_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("chunking")
        types = config.get("chunker_types", {})
        return types.get(type_name, {})

    def get_embedding_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("embedding")
        types = config.get("embedding_types", {})
        return types.get(type_name, {})

    def get_llm_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("llm")
        types = config.get("llm_types", {})
        return types.get(type_name, {})

    def get_evaluation_config(self, type_name: str) -> Dict[str, Any]:
        config = self.get_config("evaluation")
        types = config.get("evaluation_types", {})
        return types.get(type_name, {})
    
