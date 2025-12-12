# filepath: d:\開発\noto\llamaindex-experiments-main\llamaindex-experiments-main\src\db_r\database_manager.py

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# src/infrastructure/relational_stores/manager.py
"""リレーショナルストア管理クラス"""

import logging
from typing import Optional, Literal
from dataclasses import dataclass
from .sqlserver_client import SQLServerClient
from .sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


@dataclass
class RelationalStoreConfig:
    """リレーショナルストア設定"""
    db_type: Literal["sqlserver", "sqlite"] = "sqlserver"
    
    # SQL Server用
    server: Optional[str] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    port: int = 1433
    
    # SQLite用
    database_path: Optional[str] = None


class RelationalStoreManager:
    """リレーショナルストアの管理を担当"""
    
    def __init__(self, config: RelationalStoreConfig):
        self.config = config
        self._client = None
    
    def get_client(self):
        """データベースクライアントを取得"""
        if self._client is None:
            if self.config.db_type == "sqlserver":
                if not all([self.config.server, self.config.database, 
                           self.config.username, self.config.password]):
                    raise ValueError("SQL Server configuration is incomplete")
                
                self._client = SQLServerClient(
                    server=self.config.server,
                    database=self.config.database,
                    username=self.config.username,
                    password=self.config.password,
                    port=self.config.port
                )
            elif self.config.db_type == "sqlite":
                if not self.config.database_path:
                    raise ValueError("SQLite database_path is required")
                
                self._client = SQLiteClient(
                    database_path=self.config.database_path
                )
            else:
                raise ValueError(f"Unsupported db_type: {self.config.db_type}")
        
        return self._client
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.dispose()
            self._client = None

