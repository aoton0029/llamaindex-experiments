import pandas as pd
from sqlalchemy import text
from typing import Optional, Dict, Any, List
import logging

from .sqlserver_client import SQLServerClient
from .manager import RelationalStoreManager

logger = logging.getLogger(__name__)


class QueryExecutor:
    """pandas を使った SQL クエリ実行クラス"
    
    # データの取得
    df = executor.read_query("SELECT * FROM users WHERE age > :age", params={"age": 20})

    # テーブルの読み込み
    df = executor.read_table("users", schema="dbo", columns=["id", "name"])

    # DataFrame の書き込み
    executor.write_dataframe(df, "new_table", if_exists="replace")
    """
    
    def __init__(self, db_manager: RelationalStoreManager):
        """
        QueryExecutor を初期化
        
        Args:
            db_manager: DatabaseManager インスタンス
        """
        self.db_manager = db_manager
    
    def read_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        parse_dates: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        SQL クエリを実行して pandas DataFrame として取得
        
        Args:
            query: 実行する SQL クエリ
            params: クエリパラメータ
            parse_dates: 日付として解析するカラム名のリスト
            chunksize: チャンクサイズ (指定するとイテレータを返す)
            **kwargs: pandas.read_sql に渡す追加パラメータ
        
        Returns:
            クエリ結果の DataFrame
        """
        try:
            if params:
                df = pd.read_sql(
                    text(query),
                    self.db_manager.get_client().engine,
                    params=params,
                    parse_dates=parse_dates,
                    chunksize=chunksize,
                    **kwargs
                )
            else:
                df = pd.read_sql(
                    query,
                    self.db_manager.get_client().engine,
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
        """
        テーブルからデータを読み込む
        
        Args:
            table_name: テーブル名
            schema: スキーマ名
            columns: 取得するカラムのリスト (None の場合は全カラム)
            where_clause: WHERE 句 (例: "age > :min_age")
            params: WHERE 句のパラメータ
            parse_dates: 日付として解析するカラム名のリスト
            **kwargs: pandas.read_sql に渡す追加パラメータ
        
        Returns:
            テーブルデータの DataFrame
        """
        # クエリを構築
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
        """
        DataFrame をデータベーステーブルに書き込む
        
        Args:
            df: 書き込む DataFrame
            table_name: テーブル名
            schema: スキーマ名
            if_exists: テーブルが存在する場合の動作 ('fail', 'replace', 'append')
            index: DataFrame のインデックスを書き込むか
            chunksize: 一度に書き込む行数
            method: 挿入方法 (None, 'multi', callable)
            **kwargs: pandas.to_sql に渡す追加パラメータ
        
        Returns:
            書き込まれた行数
        """
        try:
            row_count = df.to_sql(
                table_name,
                self.db_manager.get_client().engine,
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
    
    def execute_stored_procedure(
        self,
        proc_name: str,
        params: Optional[Dict[str, Any]] = None,
        schema: Optional[str] = None
    ) -> pd.DataFrame:
        """
        ストアドプロシージャを実行して結果を取得
        
        Args:
            proc_name: ストアドプロシージャ名
            params: パラメータ
            schema: スキーマ名
        
        Returns:
            実行結果の DataFrame
        """
        # ストアドプロシージャの呼び出しクエリを構築
        if schema:
            full_proc_name = f"[{schema}].[{proc_name}]"
        else:
            full_proc_name = f"[{proc_name}]"
        
        if params:
            param_list = ", ".join([f":{key}" for key in params.keys()])
            query = f"EXEC {full_proc_name} {param_list}"
        else:
            query = f"EXEC {full_proc_name}"
        
        return self.read_query(query, params=params)
    
    def bulk_insert(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
        batch_size: int = 1000
    ):
        """
        DataFrame を高速バルクインサート
        
        Args:
            df: 挿入する DataFrame
            table_name: テーブル名
            schema: スキーマ名
            batch_size: バッチサイズ
        """
        try:
            # チャンクに分割して挿入
            self.write_dataframe(
                df,
                table_name,
                schema=schema,
                if_exists='append',
                index=False,
                chunksize=batch_size,
                method='multi'
            )
            logger.info(f"Bulk insert completed: {len(df)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            raise
    
    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> pd.DataFrame:
        """
        テーブルの情報を取得
        
        Args:
            table_name: テーブル名
            schema: スキーマ名
        
        Returns:
            テーブル情報の DataFrame
        """
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table_name
        """
        
        params = {"table_name": table_name}
        
        if schema:
            query += " AND TABLE_SCHEMA = :schema"
            params["schema"] = schema
        
        query += " ORDER BY ORDINAL_POSITION"
        
        return self.read_query(query, params=params)
    
    def get_table_count(self, table_name: str, schema: Optional[str] = None) -> int:
        """
        テーブルの行数を取得
        
        Args:
            table_name: テーブル名
            schema: スキーマ名
        
        Returns:
            行数
        """
        if schema:
            full_table_name = f"[{schema}].[{table_name}]"
        else:
            full_table_name = f"[{table_name}]"
        
        query = f"SELECT COUNT(*) as count FROM {full_table_name}"
        df = self.read_query(query)
        return df.iloc[0]['count']


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
