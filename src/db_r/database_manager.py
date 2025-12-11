# filepath: d:\開発\noto\llamaindex-experiments-main\llamaindex-experiments-main\src\db_r\database_manager.py

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQL Server データベース接続を管理するクラス"""
    
    def __init__(
        self,
        server: str,
        database: str,
        username: str,
        password: str,
        port: int = 1433,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        **kwargs
    ):
        """
        DatabaseManager を初期化
        
        Args:
            server: SQL Server のホスト名または IP アドレス
            database: データベース名
            username: ユーザー名
            password: パスワード
            port: ポート番号 (デフォルト: 1433)
            pool_size: コネクションプールのサイズ
            max_overflow: プールサイズを超えて作成できる接続数
            pool_timeout: 接続を待つタイムアウト時間(秒)
            **kwargs: その他の SQLAlchemy エンジンパラメータ
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.port = port
        
        # 接続文字列を構築
        self.connection_string = self._build_connection_string()
        
        # エンジンを作成
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.engine_kwargs = kwargs
        
        logger.info(f"DatabaseManager initialized for {server}/{database}")
    
    def _build_connection_string(self) -> str:
        """SQL Server 接続文字列を構築"""
        # pymssql を使用
        # パスワードに特殊文字が含まれる場合のためにURLエンコード
        encoded_username = quote_plus(self.username)
        encoded_password = quote_plus(self.password)
        conn_str = (
            f"mssql+pymssql://{encoded_username}:{encoded_password}@"
            f"{self.server}:{self.port}/{self.database}"
        )
        return conn_str
    
    @property
    def engine(self) -> Engine:
        """SQLAlchemy エンジンを取得 (遅延初期化)"""
        if self._engine is None:
            self._engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,  # 接続の有効性を確認
                echo=False,  # SQL ログを出力しない (必要に応じて True に)
                **self.engine_kwargs
            )
            logger.info("Database engine created")
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """セッションファクトリーを取得"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    @contextmanager
    def get_session(self) -> Session:
        """
        データベースセッションのコンテキストマネージャー
        
        使用例:
            with db_manager.get_session() as session:
                result = session.execute(text("SELECT * FROM table"))
        """
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
        """
        データベース接続のコンテキストマネージャー
        
        使用例:
            with db_manager.get_connection() as conn:
                result = conn.execute(text("SELECT * FROM table"))
        """
        conn = self.engine.connect()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        クエリを実行して結果を返す
        
        Args:
            query: 実行する SQL クエリ
            params: クエリパラメータ (名前付きパラメータ)
        
        Returns:
            クエリ実行結果
        """
        with self.get_connection() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return result
    
    def execute_non_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """
        クエリを実行して影響を受けた行数を返す (INSERT, UPDATE, DELETE)
        
        Args:
            query: 実行する SQL クエリ
            params: クエリパラメータ
        
        Returns:
            影響を受けた行数
        """
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
        """
        データベース接続をテスト
        
        Returns:
            接続が成功した場合 True
        """
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
        """コンテキストマネージャーのエントリー"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.dispose()
