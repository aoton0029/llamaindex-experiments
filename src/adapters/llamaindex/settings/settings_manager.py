

from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import logging

from .settings_llm import DomainLLMSettings
from .settings_template_prompts import TemplatePromptSettings

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    LLMとテンプレートプロンプトの設定を一元管理するクラス
    
    責務:
    - 設定ファイル(YAML)の読み込み
    - DomainLLMSettingsとTemplatePromptSettingsの初期化
    - 設定の検証とエラーハンドリング
    - グローバルな設定状態の管理
    """
    
    _instance: Optional['SettingsManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """シングルトンパターン"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._llm_config: Optional[Dict[str, Any]] = None
        self._template_config: Optional[Dict[str, Any]] = None
        self._config_dir: Optional[Path] = None
        
    def initialize(
        self,
        llm_model_name: str,
        config_dir: Optional[str] = None,
        llm_config_file: str = "llms.yaml",
        template_config_file: str = "templates.yaml"
    ):
        """
        設定マネージャーを初期化
        
        Args:
            llm_model_name: 使用するLLMモデル名（llms.yamlのキー）
            config_dir: 設定ファイルのディレクトリパス（Noneの場合は現在のディレクトリ）
            llm_config_file: LLM設定ファイル名
            template_config_file: テンプレート設定ファイル名
        """
        try:
            # 設定ディレクトリの解決
            if config_dir is None:
                self._config_dir = Path(__file__).parent
            else:
                self._config_dir = Path(config_dir)
            
            # 設定ファイルの読み込み
            llm_config_path = self._config_dir / llm_config_file
            template_config_path = self._config_dir / template_config_file
            
            self._llm_config = self._load_yaml(llm_config_path)
            self._template_config = self._load_yaml(template_config_path)
            
            # LLM設定の検証と初期化
            self._validate_llm_config(llm_model_name)
            self._initialize_llm_settings(llm_model_name, llm_config_path)
            
            # テンプレート設定の初期化
            self._initialize_template_settings()
            
            self._initialized = True
            logger.info(f"SettingsManager initialized with model: {llm_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SettingsManager: {e}")
            raise
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """YAMLファイルを読み込む"""
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Empty or invalid YAML file: {file_path}")
        
        logger.debug(f"Loaded config from: {file_path}")
        return config
    
    def _validate_llm_config(self, llm_model_name: str):
        """LLM設定の検証"""
        if "llm_config_models" not in self._llm_config:
            raise ValueError("Missing 'llm_config_models' in LLM config")
        
        if llm_model_name not in self._llm_config["llm_config_models"]:
            available = list(self._llm_config["llm_config_models"].keys())
            raise ValueError(
                f"LLM model '{llm_model_name}' not found. "
                f"Available models: {available}"
            )
        
        model_config = self._llm_config["llm_config_models"][llm_model_name]
        required_keys = ["backend", "model_name", "base_url"]
        missing = [k for k in required_keys if k not in model_config]
        
        if missing:
            raise ValueError(
                f"Missing required keys in model config: {missing}"
            )
    
    def _initialize_llm_settings(self, llm_model_name: str):
        """DomainLLMSettingsを初期化"""
        # llm_config_modelsから選択したモデルの設定を取得
        model_config = self._llm_config["llm_config_models"][llm_model_name]
        
        # llm_domain_kwargs_modelsを取得
        domain_kwargs = self._llm_config.get("llm_domain_kwargs_models", {})
        
        DomainLLMSettings.initialize(model_config, domain_kwargs)
        
        logger.info(f"DomainLLMSettings initialized for model: {llm_model_name}")
    
    def _initialize_template_settings(self):
        """TemplatePromptSettingsを初期化"""
        template_prompts = self._template_config.get("template_prompts", {})
        TemplatePromptSettings.initialize(template_prompts)
        logger.info("TemplatePromptSettings initialized")
    
    @property
    def llm_settings(self):
        """DomainLLMSettingsへのアクセス"""
        if not self._initialized:
            raise RuntimeError("SettingsManager not initialized. Call initialize() first.")
        return DomainLLMSettings
    
    @property
    def template_settings(self):
        """TemplatePromptSettingsへのアクセス"""
        if not self._initialized:
            raise RuntimeError("SettingsManager not initialized. Call initialize() first.")
        return TemplatePromptSettings
    
    def get_status(self) -> Dict[str, Any]:
        """現在の設定状態を取得"""
        return {
            "initialized": self._initialized,
            "config_dir": str(self._config_dir) if self._config_dir else None,
            "llm_models": list(self._llm_config.get("llm_config_models", {}).keys()) if self._llm_config else [],
            "domain_kwargs_count": len(self._llm_config.get("llm_domain_kwargs_models", {})) if self._llm_config else 0,
            "template_count": len(self._template_config.get("template_prompts", {})) if self._template_config else 0,
        }
    
    def reload(
        self,
        llm_model_name: Optional[str] = None,
        reload_llm: bool = True,
        reload_templates: bool = True
    ):
        """設定を再読み込み"""
        if not self._initialized:
            raise RuntimeError("Cannot reload before initialization")
        
        try:
            if reload_llm and llm_model_name:
                llm_config_path = self._config_dir / "llms.yaml"
                self._llm_config = self._load_yaml(llm_config_path)
                self._initialize_llm_settings(llm_model_name, llm_config_path)
            
            if reload_templates:
                template_config_path = self._config_dir / "templates.yaml"
                self._template_config = self._load_yaml(template_config_path)
                self._initialize_template_settings()
            
            logger.info("Settings reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload settings: {e}")
            raise


# グローバルインスタンス
settings_manager = SettingsManager()