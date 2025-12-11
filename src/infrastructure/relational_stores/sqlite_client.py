# src/infrastructure/relational_stores/sqlite_client.py
"""SQLite クライアントクラス"""

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class SQLiteClient:
    """SQLite データベース接続を管理するクラス"""
    
    def __init__(
        self,
        database_path: str,
        check_same_thread: bool = False,
        **kwargs
    ):
        self.database_path = database_path
        self.check_same_thread = check_same_thread
        
        # データベースディレクトリが存在しない場合は作成
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.connection_string = f"sqlite:///{database_path}"
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self.engine_kwargs = kwargs
        
        logger.info(f"SQLiteClient initialized for {database_path}")
    
    @property
    def engine(self) -> Engine:
        """SQLAlchemy エンジンを取得 (遅延初期化)"""
        if self._engine is None:
            connect_args = {"check_same_thread": self.check_same_thread}
            self._engine = create_engine(
                self.connection_string,
                connect_args=connect_args,
                echo=False,
                **self.engine_kwargs
            )
            logger.info("SQLite engine created")
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """セッションファクトリーを取得"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    @contextmanager
    def get_session(self) -> Session:
        """データベースセッションのコンテキストマネージャー"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self):
        """データベース接続のコンテキストマネージャー"""
        conn = self.engine.connect()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """クエリを実行して結果を返す"""
        with self.get_connection() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return result
    
    def execute_non_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """クエリを実行して影響を受けた行数を返す"""
        with self.get_connection() as conn:
            trans = conn.begin()
            try:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                trans.commit()
                return result.rowcount
            except Exception as e:
                trans.rollback()
                logger.error(f"Execute non-query error: {e}")
                raise
    
    def test_connection(self) -> bool:
        """データベース接続をテスト"""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def dispose(self):
        """接続プールを破棄"""
        if self._engine is not None:
            self._engine.dispose()
            logger.info("Database engine disposed")
            self._engine = None
            self._session_factory = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()