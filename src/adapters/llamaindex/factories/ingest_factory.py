from typing import List
import logging
from llama_index.core.schema import Document, BaseNode
from llama_index.core.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

class PipelineFactory:
    """
    目的:
        生データ（ファイル、テキスト、URL 等）を受け取り、検索／検索用インデックスに格納できる「ノード（チャンク）＋メタデータ＋埋め込み」の形に変換して保存すること。
        読み込み→分割→前処理→メタデータ抽出→ストア保存までの一連の処理を一元的にオーケストレーションする。

    構成要素:
        - readers: ドキュメント読み込み器（ファイル、ディレクトリ、API など）
        - node_parser / chunker: テキストを意味あるチャンク（BaseNode）に分割
        - transformations: クリーニングや正規化などの前処理パイプ
        - extractors: 各ノードからタイトル／要約／キーワード等を抽出して metadata を作る
        - docstore: ドキュメント原本やメタ情報の保持（検索や参照用）
        - vector_store: 埋め込みベクトルの格納（Milvus 等）
        - docstore_strategy: docstore と vector_store の同期方法などを制御

    用途:
        1. readers で Document を読み込む。  
        2. node_parser で Document → BaseNode（チャンク）に変換。  
        3. transformations でノードを整形・正規化。  
        4. extractors でタイトル／要約／キーワード等を node.metadata として付与。  
        5. vector_store に埋め込みを作成して保存、必要なら docstore に原本やメタ情報を保存。  
        6. docstore_strategy に従って ID／メタを整合させる。  
        7. 保存済みデータは検索・回答生成パイプライン（retriever / synthsizer）で利用される。
    """
    @staticmethod
    def create(ingest_type:str, **kwargs) -> IngestionPipeline:
        if ingest_type == "default":
            return PipelineFactory.create_default_pipeline()
        else:
            raise ValueError(f"未知のパイプラインタイプ: {ingest_type}")
    
    @staticmethod
    def create_default_pipeline() -> IngestionPipeline:
        
        pipeline = IngestionPipeline(
            # node_parser=,
            # transformations=,
            # readers=,
            # documents=,
            # vector_store=,
            # docstore=,
            # docstore_strategy=,
            # extractors=
        )
        return pipeline
    
