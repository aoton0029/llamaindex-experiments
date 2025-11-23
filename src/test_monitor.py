import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from llama_index.core import Settings, StorageContext

logger = logging.getLogger(__name__)

class TestMetrics(BaseModel):
    """実験メトリクス"""
    chunking_time: float = 0.0
    indexing_time: float = 0.0
    total_documents: int = 0
    total_nodes: int = 0
    index_size_bytes: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    llm_tokens_used: Optional[int] = None
    embedding_tokens_used: Optional[int] = None

class TestMonitor:
    """実験のモニタリングと記録を管理"""
    
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment_id: Optional[str] = None
        self.current_experiment_dir: Optional[Path] = None
        self.metrics = TestMetrics()
        self.start_time: Optional[datetime] = None
        self.logs: List[Dict[str, Any]] = []
    
    def start_test(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        実験を開始し、ディレクトリ構造を作成
        
        Args:
            experiment_name: 実験名
            config: 実験設定
            
        Returns:
            実験ID
        """
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.current_experiment_id = f"{experiment_name}_{timestamp}"
        self.current_experiment_dir = self.result_dir / self.current_experiment_id
        
        # ディレクトリ構造を作成
        (self.current_experiment_dir / "config").mkdir(parents=True, exist_ok=True)
        (self.current_experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.current_experiment_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.current_experiment_dir / "results").mkdir(parents=True, exist_ok=True)
        
        # 設定を保存
        self._save_config(config)
        
        # メタデータを保存
        metadata = {
            "experiment_id": self.current_experiment_id,
            "experiment_name": experiment_name,
            "start_time": self.start_time.isoformat(),
            "status": "running"
        }
        self._save_json(metadata, self.current_experiment_dir / "metadata.json")
        
        logger.info(f"Started experiment: {self.current_experiment_id}")
        return self.current_experiment_id
    
    def log_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """イベントをログに記録"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        self.logs.append(event)
        logger.info(f"[{event_type}] {message}")
    
    def update_metrics(self, **kwargs):
        """メトリクスを更新"""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def end_test(self, success: bool, result_data: Dict[str, Any]):
        """
        実験を終了し、最終結果を保存
        
        Args:
            success: 実験が成功したか
            result_data: 結果データ
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # 最終メタデータを更新
        metadata = {
            "experiment_id": self.current_experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "status": "completed" if success else "failed",
            "success": success
        }
        self._save_json(metadata, self.current_experiment_dir / "metadata.json")
        
        # ログを保存
        self._save_logs()
        
        # メトリクスを保存
        self._save_metrics()
        
        # 結果を保存
        self._save_results(result_data)
        
        # サマリーを作成
        self._create_summary(success, duration)
        
        logger.info(f"Ended experiment: {self.current_experiment_id} (Duration: {duration:.2f}s)")
    
    def _save_config(self, config: Dict[str, Any]):
        """設定を保存"""
        config_path = self.current_experiment_dir / "config" / "experiment_config.json"
        self._save_json(config, config_path)
    
    def _save_logs(self):
        """ログを保存"""
        logs_path = self.current_experiment_dir / "logs" / "events.json"
        self._save_json(self.logs, logs_path)
        
        # CSV形式でも保存（分析用）
        if self.logs:
            df = pd.DataFrame(self.logs)
            csv_path = self.current_experiment_dir / "logs" / "events.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    def _save_metrics(self):
        """メトリクスを保存"""
        metrics_dict = self.metrics.model_dump()
        metrics_path = self.current_experiment_dir / "metrics" / "metrics.json"
        self._save_json(metrics_dict, metrics_path)
        
        # CSV形式でも保存
        df = pd.DataFrame([metrics_dict])
        csv_path = self.current_experiment_dir / "metrics" / "metrics.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    def _save_results(self, result_data: Dict[str, Any]):
        """結果を保存"""
        results_path = self.current_experiment_dir / "results" / "results.json"
        self._save_json(result_data, results_path)
    
    def _create_summary(self, success: bool, duration: float):
        """サマリーを作成"""
        summary = {
            "experiment_id": self.current_experiment_id,
            "success": success,
            "duration_seconds": duration,
            "metrics": self.metrics.model_dump(),
            "event_count": len(self.logs),
            "final_status": "completed" if success else "failed"
        }
        
        summary_path = self.current_experiment_dir / "summary.json"
        self._save_json(summary, summary_path)
        
        # 人間が読みやすいテキスト形式でも保存
        self._save_summary_text(summary)
    
    def _save_summary_text(self, summary: Dict[str, Any]):
        """サマリーをテキスト形式で保存"""
        summary_text_path = self.current_experiment_dir / "summary.txt"
        
        with open(summary_text_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"実験ID: {summary['experiment_id']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"結果: {'成功' if summary['success'] else '失敗'}\n")
            f.write(f"実行時間: {summary['duration_seconds']:.2f}秒\n")
            f.write(f"イベント数: {summary['event_count']}\n\n")
            
            f.write("メトリクス:\n")
            f.write("-" * 40 + "\n")
            for key, value in summary['metrics'].items():
                f.write(f"  {key}: {value}\n")
    
    def _save_json(self, data: Any, path: Path):
        """JSONファイルを保存"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_test_dir(self) -> Path:
        """現在の実験ディレクトリを取得"""
        return self.current_experiment_dir

