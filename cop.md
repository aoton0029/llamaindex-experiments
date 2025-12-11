aoton0029: llamaindexのStorageContext用にmilvus、redis、mongodb、neo4jのクライアントを管理している #file:db と sqlserverやsqliteのクライアントを管理している #file:db_r がある。これらはどのような構成・レイヤー・フォルダ分け・ネームスペースにしたらいい？

GitHub Copilot: 2つのフォルダの内容を確認して、適切な構成を提案します。 


## 推奨構成

### 1. **用途別レイヤー分け**

```
src/
  storage/                      # LlamaIndexのStorageContext関連
    __init__.py
    nosql/                      # NoSQL & ベクトルDB
      __init__.py
      clients/
        milvus_client.py
        mongodb_client.py
        neo4j_client.py
        redis_client.py
      manager.py                # 旧database_manager.py
      storage_context_manager.py
    
  persistence/                  # リレーショナルDB (永続化層)
    __init__.py
    rdb/                        # RDB汎用
      __init__.py
      clients/
        sqlserver_client.py     # 旧database_manager.py
        sqlite_client.py
      manager.py
      query_executor.py         # 旧query.py
```

### 2. **機能別レイヤー分け（Alternative）**

```
src/
  infrastructure/               # インフラストラクチャ層
    __init__.py
    databases/
      __init__.py
      nosql/                   # NoSQL系 (LlamaIndex用)
        __init__.py
        milvus_client.py
        mongodb_client.py
        neo4j_client.py
        redis_client.py
        manager.py
        
      relational/              # RDB系
        __init__.py
        sqlserver_client.py
        sqlite_client.py
        manager.py
        query_executor.py
        
    llamaindex/                # LlamaIndex固有
      __init__.py
      storage_context_manager.py
```

### 3. **推奨：ドメイン駆動設計アプローチ**

```
src/
  infrastructure/               # インフラストラクチャ層
    __init__.py
    
    vector_stores/             # ベクトルストア
      __init__.py
      milvus_client.py
      manager.py
      
    document_stores/           # ドキュメントストア
      __init__.py
      mongodb_client.py
      manager.py
      
    graph_stores/              # グラフストア
      __init__.py
      neo4j_client.py
      manager.py
      
    index_stores/              # インデックスストア
      __init__.py
      redis_client.py
      manager.py
      
    relational_stores/         # リレーショナルストア
      __init__.py
      sqlserver_client.py
      sqlite_client.py
      manager.py
      query_executor.py
      
  adapters/
    llamaindex/
      storage/
        __init__.py
        storage_context_manager.py  # 各インフラを統合
```

## ネームスペースの提案

現在の用途に基づくと、**オプション1（用途別）が最も適切**です：

```python
# LlamaIndex StorageContext用
from src.storage.nosql.manager import DatabaseManager as NoSQLManager
from src.storage.nosql.clients.milvus_client import MilvusClient
from src.storage.nosql.storage_context_manager import StorageContextManager

# リレーショナルDB用（分析・クエリ用途）
from src.persistence.rdb.manager import DatabaseManager as RDBManager
from src.persistence.rdb.query_executor import QueryExecutor
```

## 主な理由

1. **明確な責務分離**: `db`はLlamaIndexのStorageContext管理、`db_r`は一般的なSQL実行
2. **命名の明確化**: `storage`（RAG用）vs `persistence`（永続化層）
3. **スケーラビリティ**: 将来的に他のDBやストレージが追加されやすい
4. **依存関係の明確化**: LlamaIndex依存は`storage`配下に集約

どの構成が良いか、またはカスタマイズが必要な場合は教えてください。

aoton0029: 3のアプローチで各ファイル

GitHub Copilot: オプション3のドメイン駆動設計アプローチで各ファイルを移行・整理します。 

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db/milvus_client.py#1-1), lines 1 to 150

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db/mongodb_client.py#1-1), lines 1 to 150

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db/neo4j_client.py#1-1), lines 1 to 150

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db/redis_client.py#1-1), lines 1 to 150

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db/database_manager.py)

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db/storage_context_manager.py)

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db_r/database_manager.py)

Read [](file:///d%3A/notoa/Documents/PythonProjects/llamaindex-experiments/src/db_r/query.py)

すべてのファイルを確認しました。オプション3のドメイン駆動設計アプローチで新しいフォルダ構成とファイルを作成します。

## 新しいフォルダ構成

```
src/
  infrastructure/
    vector_stores/
      __init__.py
      milvus_client.py
      manager.py
    document_stores/
      __init__.py
      mongodb_client.py
      manager.py
    graph_stores/
      __init__.py
      neo4j_client.py
      manager.py
    index_stores/
      __init__.py
      redis_client.py
      manager.py
    relational_stores/
      __init__.py
      sqlserver_client.py
      sqlite_client.py
      manager.py
      query_executor.py
  adapters/
    llamaindex/
      storage/
        __init__.py
        config.py
        storage_context_manager.py
```

以下のコードブロックで各ファイルの内容を示します：

### 1. インフラストラクチャ層 - ベクトルストア

```python
# src/infrastructure/vector_stores/__init__.py
from .milvus_client import MilvusClient
from .manager import VectorStoreManager, VectorStoreConfig

__all__ = [
    "MilvusClient",
    "VectorStoreManager",
    "VectorStoreConfig",
]
```

```python
# src/infrastructure/vector_stores/milvus_client.py
# (既存の milvus_client.py をそのまま移動)
```

```python
# src/infrastructure/vector_stores/manager.py
"""ベクトルストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.vector_stores.types import VectorStore
from .milvus_client import MilvusClient

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """ベクトルストア設定"""
    host: str = "milvus"
    port: int = 19530
    user: Optional[str] = "admin"
    password: Optional[str] = "pdntsPa0"


class VectorStoreManager:
    """ベクトルストアの管理を担当"""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._client: Optional[MilvusClient] = None
    
    def get_client(self) -> MilvusClient:
        """Milvusクライアントを取得"""
        if self._client is None:
            self._client = MilvusClient(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password
            )
        return self._client
    
    def get_vector_store(
        self,
        collection_name: str,
        dim: int,
        **kwargs
    ) -> VectorStore:
        """ベクトルストアを取得"""
        client = self.get_client()
        return client.get_vector_store(
            collection_name=collection_name,
            dim=dim,
            **kwargs
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None
```

### 2. インフラストラクチャ層 - ドキュメントストア

```python
# src/infrastructure/document_stores/__init__.py
from .mongodb_client import MongoDBClient
from .manager import DocumentStoreManager, DocumentStoreConfig

__all__ = [
    "MongoDBClient",
    "DocumentStoreManager",
    "DocumentStoreConfig",
]
```

```python
# src/infrastructure/document_stores/mongodb_client.py
# (既存の mongodb_client.py をそのまま移動)
```

```python
# src/infrastructure/document_stores/manager.py
"""ドキュメントストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.storage.docstore.types import BaseDocumentStore
from .mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


@dataclass
class DocumentStoreConfig:
    """ドキュメントストア設定"""
    host: str = "mongodb"
    port: int = 27017
    database_name: str = "pdf_rag_system"
    username: Optional[str] = "admin"
    password: Optional[str] = "pdntsPa0"
    auth_source: str = "admin"


class DocumentStoreManager:
    """ドキュメントストアの管理を担当"""
    
    def __init__(self, config: Optional[DocumentStoreConfig] = None):
        self.config = config or DocumentStoreConfig()
        self._client: Optional[MongoDBClient] = None
    
    def get_client(self) -> MongoDBClient:
        """MongoDBクライアントを取得"""
        if self._client is None:
            self._client = MongoDBClient(
                host=self.config.host,
                port=self.config.port,
                database_name=self.config.database_name,
                username=self.config.username,
                password=self.config.password,
                auth_source=self.config.auth_source
            )
        return self._client
    
    def get_docstore(
        self,
        namespace: str = "default",
        collection_name: str = "documents"
    ) -> BaseDocumentStore:
        """ドキュメントストアを取得"""
        client = self.get_client()
        return client.get_docstore(
            namespace=namespace,
            collection_name=collection_name
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None
```

### 3. インフラストラクチャ層 - グラフストア

```python
# src/infrastructure/graph_stores/__init__.py
from .neo4j_client import Neo4jClient
from .manager import GraphStoreManager, GraphStoreConfig

__all__ = [
    "Neo4jClient",
    "GraphStoreManager",
    "GraphStoreConfig",
]
```

```python
# src/infrastructure/graph_stores/neo4j_client.py
# (既存の neo4j_client.py をそのまま移動)
```

```python
# src/infrastructure/graph_stores/manager.py
"""グラフストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.graph_stores.types import GraphStore
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class GraphStoreConfig:
    """グラフストア設定"""
    uri: str = "bolt://neo4j:7687"
    username: str = "neo4j"
    password: str = "pdntsPa0"
    database: str = "neo4j"


class GraphStoreManager:
    """グラフストアの管理を担当"""
    
    def __init__(self, config: Optional[GraphStoreConfig] = None):
        self.config = config or GraphStoreConfig()
        self._client: Optional[Neo4jClient] = None
    
    def get_client(self) -> Neo4jClient:
        """Neo4jクライアントを取得"""
        if self._client is None:
            self._client = Neo4jClient(
                uri=self.config.uri,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database
            )
        return self._client
    
    def get_graph_store(
        self,
        node_label: str = "Entity",
        rel_type: str = "RELATED",
        **kwargs
    ) -> GraphStore:
        """グラフストアを取得"""
        client = self.get_client()
        return client.get_graph_store(
            node_label=node_label,
            rel_type=rel_type,
            **kwargs
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None
```

### 4. インフラストラクチャ層 - インデックスストア

```python
# src/infrastructure/index_stores/__init__.py
from .redis_client import RedisClient
from .manager import IndexStoreManager, IndexStoreConfig

__all__ = [
    "RedisClient",
    "IndexStoreManager",
    "IndexStoreConfig",
]
```

```python
# src/infrastructure/index_stores/redis_client.py
# (既存の redis_client.py をそのまま移動)
```

```python
# src/infrastructure/index_stores/manager.py
"""インデックスストア管理クラス"""

import logging
from typing import Optional
from dataclasses import dataclass
from llama_index.core.storage.index_store.types import BaseIndexStore
from .redis_client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class IndexStoreConfig:
    """インデックスストア設定"""
    host: str = "redis-store"
    port: int = 6379
    password: Optional[str] = "pdntsPa0"
    db: int = 0


class IndexStoreManager:
    """インデックスストアの管理を担当"""
    
    def __init__(self, config: Optional[IndexStoreConfig] = None):
        self.config = config or IndexStoreConfig()
        self._client: Optional[RedisClient] = None
    
    def get_client(self) -> RedisClient:
        """Redisクライアントを取得"""
        if self._client is None:
            self._client = RedisClient(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db
            )
        return self._client
    
    def get_index_store(
        self,
        namespace: str = "default",
        collection_suffix: str = ""
    ) -> BaseIndexStore:
        """インデックスストアを取得"""
        client = self.get_client()
        return client.get_index_store(
            namespace=namespace,
            collection_suffix=collection_suffix
        )
    
    def disconnect(self) -> None:
        """接続を切断"""
        if self._client:
            self._client.disconnect()
            self._client = None
```

### 5. インフラストラクチャ層 - リレーショナルストア

```python
# src/infrastructure/relational_stores/__init__.py
from .sqlserver_client import SQLServerClient
from .sqlite_client import SQLiteClient
from .manager import RelationalStoreManager, RelationalStoreConfig
from .query_executor import QueryExecutor, create_query_executor

__all__ = [
    "SQLServerClient",
    "SQLiteClient",
    "RelationalStoreManager",
    "RelationalStoreConfig",
    "QueryExecutor",
    "create_query_executor",
]
```

```python
# src/infrastructure/relational_stores/sqlserver_client.py
"""SQL Server クライアントクラス"""

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class SQLServerClient:
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
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.port = port
        
        self.connection_string = self._build_connection_string()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.engine_kwargs = kwargs
        
        logger.info(f"SQLServerClient initialized for {server}/{database}")
    
    def _build_connection_string(self) -> str:
        """SQL Server 接続文字列を構築"""
        encoded_username = quote_plus(self.username)
        encoded_password = quote_plus(self.password)
        return (
            f"mssql+pymssql://{encoded_username}:{encoded_password}@"
            f"{self.server}:{self.port}/{self.database}"
        )
    
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
                pool_pre_ping=True,
                echo=False,
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
```

```python
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
```

```python
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
```

```python
# src/infrastructure/relational_stores/query_executor.py
# (既存の query.py の内容をそのまま移動し、DatabaseManager を SQLServerClient/SQLiteClient に対応)

import pandas as pd
from sqlalchemy import text
from typing import Optional, Dict, Any, List, Union
import logging
from .sqlserver_client import SQLServerClient
from .sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


class QueryExecutor:
    """pandas を使った SQL クエリ実行クラス"""
    
    def __init__(self, db_client: Union[SQLServerClient, SQLiteClient]):
        self.db_client = db_client
    
    def read_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        parse_dates: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """SQL クエリを実行して pandas DataFrame として取得"""
        try:
            if params:
                df = pd.read_sql(
                    text(query),
                    self.db_client.engine,
                    params=params,
                    parse_dates=parse_dates,
                    chunksize=chunksize,
                    **kwargs
                )
            else:
                df = pd.read_sql(
                    query,
                    self.db_client.engine,
                    parse_dates=parse_dates,
                    chunksize=chunksize,
                    **kwargs
                )
            logger.info(f"Query executed successfully, returned {len(df) if chunksize is None else 'chunked'} rows")
            return df
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise
    
    def read_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        parse_dates: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """テーブルからデータを読み込む"""
        cols = ", ".join(columns) if columns else "*"
        
        if schema:
            full_table_name = f"[{schema}].[{table_name}]"
        else:
            full_table_name = f"[{table_name}]"
        
        query = f"SELECT {cols} FROM {full_table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        return self.read_query(query, params=params, parse_dates=parse_dates, **kwargs)
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = 'append',
        index: bool = False,
        chunksize: Optional[int] = None,
        method: Optional[str] = None,
        **kwargs
    ) -> int:
        """DataFrame をデータベーステーブルに書き込む"""
        try:
            row_count = df.to_sql(
                table_name,
                self.db_client.engine,
                schema=schema,
                if_exists=if_exists,
                index=index,
                chunksize=chunksize,
                method=method,
                **kwargs
            )
            logger.info(f"DataFrame written to {table_name}, {row_count} rows")
            return row_count if row_count else len(df)
        except Exception as e:
            logger.error(f"DataFrame write error: {e}")
            raise
    
    # ... (残りのメソッドも同様に移行)


def create_query_executor(
    server: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> QueryExecutor:
    """QueryExecutor を簡単に作成するヘルパー関数"""
    db_client = SQLServerClient(
        server=server,
        database=database,
        username=username,
        password=password,
        **kwargs
    )
    return QueryExecutor(db_client)
```

### 6. アダプター層 - LlamaIndex StorageContext

```python
# src/adapters/llamaindex/storage/__init__.py
from .config import (
    DocstoreConfig,
    IndexStoreConfig,
    VectorStoreConfig,
    ImageStoreConfig,
    GraphStoreConfig,
    StorageContextConfig,
)
from .storage_context_manager import StorageContextManager

__all__ = [
    "DocstoreConfig",
    "IndexStoreConfig",
    "VectorStoreConfig",
    "ImageStoreConfig",
    "GraphStoreConfig",
    "StorageContextConfig",
    "StorageContextManager",
]
```

```python
# src/adapters/llamaindex/storage/config.py
"""StorageContext関連の設定クラス"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from pymilvus import CollectionSchema


@dataclass
class DocstoreConfig:
    """ドキュメントストア設定"""
    namespace: str = "default"
    collection_name: str = "documents"


@dataclass
class IndexStoreConfig:
    """インデックスストア設定"""
    namespace: str = "default"
    collection_suffix: str = "index"


@dataclass
class VectorStoreConfig:
    """ベクトルストア設定"""
    collection_name: str
    dim: int = 8192
    schema: Optional[CollectionSchema] = None
    metric_type: str = "COSINE"
    index_type: str = "IVF_FLAT"
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class ImageStoreConfig:
    """画像ストア設定"""
    collection_name: str
    dim: int = 512
    schema: Optional[CollectionSchema] = None
    metric_type: str = "L2"
    index_type: str = "IVF_FLAT"
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class GraphStoreConfig:
    """グラフストア設定"""
    node_label: str = "Entity"
    rel_type: str = "RELATED"
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class StorageContextConfig:
    """StorageContext設定"""
    context_name: str
    docstore: Optional[DocstoreConfig] = None
    index_store: Optional[IndexStoreConfig] = None
    vector_store: Optional[VectorStoreConfig] = None
    image_store: Optional[ImageStoreConfig] = None
    graph_store: Optional[GraphStoreConfig] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StorageContextConfig':
        """辞書からStorageContextConfigを生成"""
        docstore = None
        if "docstore" in config_dict and config_dict["docstore"]:
            docstore = DocstoreConfig(**config_dict["docstore"])
        
        index_store = None
        if "index_store" in config_dict and config_dict["index_store"]:
            index_store = IndexStoreConfig(**config_dict["index_store"])
        
        vector_store = None
        if "vector_store" in config_dict and config_dict["vector_store"]:
            vector_store = VectorStoreConfig(**config_dict["vector_store"])
        
        image_store = None
        if "image_store" in config_dict and config_dict["image_store"]:
            image_store = ImageStoreConfig(**config_dict["image_store"])
        
        graph_store = None
        if "graph_store" in config_dict and config_dict["graph_store"]:
            graph_store = GraphStoreConfig(**config_dict["graph_store"])
        
        return cls(
            context_name=config_dict["context_name"],
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            image_store=image_store,
            graph_store=graph_store
        )
```

```python
# src/adapters/llamaindex/storage/storage_context_manager.py
"""StorageContextの構築と管理を担当するクラス"""

import logging
from typing import Optional, Dict, List
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage, load_indices_from_storage

from src.infrastructure.vector_stores.manager import VectorStoreManager
from src.infrastructure.document_stores.manager import DocumentStoreManager
from src.infrastructure.graph_stores.manager import GraphStoreManager
from src.infrastructure.index_stores.manager import IndexStoreManager
from .config import StorageContextConfig

logger = logging.getLogger(__name__)


class StorageContextManager:
    """StorageContextの構築と管理を担当"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        document_store_manager: DocumentStoreManager,
        index_store_manager: IndexStoreManager,
        graph_store_manager: Optional[GraphStoreManager] = None
    ):
        self.vector_store_manager = vector_store_manager
        self.document_store_manager = document_store_manager
        self.index_store_manager = index_store_manager
        self.graph_store_manager = graph_store_manager
        
        self._storage_contexts: Dict[str, StorageContext] = {}
        self._storage_configs: Dict[str, StorageContextConfig] = {}
    
    def create_storage_context(self, config: StorageContextConfig) -> StorageContext:
        """StorageContextを作成"""
        kwargs = {}
        
        if config.docstore:
            kwargs["docstore"] = self.document_store_manager.get_docstore(
                namespace=config.docstore.namespace,
            )
        
        if config.index_store:
            kwargs["index_store"] = self.index_store_manager.get_index_store(
                namespace=config.index_store.namespace,
                collection_suffix=config.index_store.collection_suffix
            )
        
        if config.vector_store:
            kwargs["vector_store"] = self.vector_store_manager.get_vector_store(
                collection_name=config.vector_store.collection_name,
                dim=config.vector_store.dim,
                schema=config.vector_store.schema,
            )
            try:
                milvus_client = self.vector_store_manager.get_client()
                milvus_client.load_collection(config.vector_store.collection_name)
                logger.info(f"Successfully loaded Milvus Collection: {config.vector_store.collection_name}")
            except Exception as e:
                logger.info(f"collection loaded Error: {config.vector_store.collection_name}")
        
        if config.image_store:
            kwargs["image_store"] = self.vector_store_manager.get_vector_store(
                collection_name=config.image_store.collection_name,
                dim=config.image_store.dim,
                schema=config.image_store.schema,
            )
        
        if config.graph_store and self.graph_store_manager:
            kwargs["graph_store"] = self.graph_store_manager.get_graph_store(
                node_label=config.graph_store.node_label,
                rel_type=config.graph_store.rel_type
            )
        
        storage_context = StorageContext.from_defaults(**kwargs)
        
        self._storage_contexts[config.context_name] = storage_context
        self._storage_configs[config.context_name] = config
        logger.info(f"StorageContext '{config.context_name}' を作成しました")
        return storage_context
    
    def get_storage_context(self, context_name: str) -> Optional[StorageContext]:
        """作成済みのStorageContextを取得"""
        return self._storage_contexts.get(context_name)
    
    def load_index(self, context_name: str, index_id: Optional[str]) -> BaseIndex | List[BaseIndex]:
        """StorageContextからインデックスをロード"""
        storage_context = self.get_storage_context(context_name)
        if not storage_context:
            raise ValueError(f"StorageContext '{context_name}' が見つかりません")
        
        index = load_index_from_storage(storage_context, index_id=index_id)
        logger.info(f"インデックス '{index_id}' をロードしました")
        return index
    
    def load_indices(self, context_name: str) -> List[BaseIndex]:
        """全インデックスをロード"""
        storage_context = self.get_storage_context(context_name)
        if not storage_context:
            raise ValueError(f"StorageContext '{context_name}' が見つかりません")
        
        indices = load_indices_from_storage(storage_context)
        logger.info(f"{len(indices)} 個のインデックスをロードしました")
        return indices
    
    # ... (その他のメソッドも同様に実装)
```

## 使用例

```python
# インフラストラクチャ層の使用例
from src.infrastructure.vector_stores import VectorStoreManager, VectorStoreConfig
from src.infrastructure.document_stores import DocumentStoreManager, DocumentStoreConfig
from src.infrastructure.index_stores import IndexStoreManager, IndexStoreConfig
from src.infrastructure.relational_stores import RelationalStoreManager, RelationalStoreConfig, QueryExecutor

# LlamaIndex用
vector_mgr = VectorStoreManager(VectorStoreConfig())
doc_mgr = DocumentStoreManager(DocumentStoreConfig())
index_mgr = IndexStoreManager(IndexStoreConfig())

# StorageContext作成
from src.adapters.llamaindex.storage import StorageContextManager, StorageContextConfig

storage_mgr = StorageContextManager(
    vector_store_manager=vector_mgr,
    document_store_manager=doc_mgr,
    index_store_manager=index_mgr
)

# リレーショナルDB用
rdb_config = RelationalStoreConfig(
    db_type="sqlserver",
    server="localhost",
    database="mydb",
    username="user",
    password="pass"
)
rdb_mgr = RelationalStoreManager(rdb_config)
executor = QueryExecutor(rdb_mgr.get_client())
df = executor.read_query("SELECT * FROM users")
```

この構成により、責務が明確に分離され、それぞれのストアが独立して管理できるようになります。