# 会話情報RAGシステム 設計書

## 1. システム目的
営業と取引先の会話情報から、取引先が何を求めているかを効率的に検索・抽出するRAGシステムを構築する。

## 2. システム構成

### 2.1 主要コンポーネント
```
[既存DB] → [データ取得層] → [前処理層] → [ベクトル化層] → [ベクトルDB/インデックスDB/ドキュメントDB]
                                                                    ↓
[ユーザー] → [クエリ処理] → [検索エンジン] ←-------------------[ベクトルDB/インデックスDB/ドキュメントDB]
                                   ↓
                            [LLMによる回答生成] → [回答出力]
```

### 2.2 技術スタック候補
- **ベクトルDB**: Milvus
- **インデックスDB**: Redis
- **ドキュメントDB**: Mongodb
- **Embedding Model**: qwen3-embedding:8b
- **LLM**: Qwen3-32B-AWQ
- **RAGフレームワーク**: LlamaIndex

## 3. データ処理設計

### 3.1 インデックス作成フロー

#### 3.1.1 データ取得
- 既存DBから文字起こし情報を取得
- 取得単位: 1会話セッション = 1レコード

#### 既存DBスキーマ
##### 会話セッション
- uid(会話セッションのユニークID)
- 営業担当者名
- 会社名
- 拠点名
- 部署名
- 取引先担当者名
- 会話情報(発話情報のリスト)
- 要約

##### 発話情報(話者不明)
- 発話開始時間(s)
- 発話終了時間(s)
- 発話内容

##### 要約
要約の構成
```
□全体概要
<内容>

□トピック別要約
Topic1: <Topic1のタイトル>
・<内容>
・<内容>
Topic2: <Topic2のタイトル>
・<内容>
・<内容>

□決定事項
・<決定事項>
・<決定事項>
```

#### 3.1.2 ドキュメント構造設計
各会話セッションを1つのDocumentとして扱う。

##### Document構成
```
Document {
  text: 会話の全テキスト（発話内容を時系列で結合）,
  metadata: {
    uid: 会話セッションのユニークID,
    営業担当者名: string,
    会社名: string,
    拠点名: string,
    部署名: string,
    取引先担当者名: string,
    要約_全体概要: string,
    要約_トピック別: string,
    要約_決定事項: string,
    会話開始時間: timestamp,
    会話終了時間: timestamp
  }
}
```

##### テキスト構成例
```
[00:15] 発話内容1
[00:32] 発話内容2
[01:05] 発話内容3
...
```

#### 3.1.3 チャンキング戦略
- **チャンクサイズ**: 512トークン
- **オーバーラップ**: 128トークン
- **チャンク単位**: 意味的なまとまりを考慮（発話の途中で分割しない）
- 各チャンクに親Documentのmetadataを継承

#### 3.1.4 Embedding生成
- **モデル**: qwen3-embedding:8b
- **ベクトル次元**: 8192次元
- **処理単位**: チャンク単位でEmbeddingを生成
- **バッチ処理**: 複数チャンクをまとめて処理して効率化

### 3.2 LlamaIndexによるインデックス管理

#### 3.2.1 StorageContext設計
全ての会話セッションを単一のStorageContextで管理する。

##### StorageContextの構成要素
```
StorageContext {
  - vector_store: MilvusVectorStore（ベクトル検索用）
  - index_store: RedisIndexStore（インデックス管理用）
  - docstore: MongoDocumentStore（元ドキュメント保存用）
}
```

#### 3.2.2 インデックス作成フロー
1. **初回作成時**
   - StorageContextを初期化（Milvus/Redis/MongoDB接続）
   - 各会話セッションをDocumentに変換
   - VectorStoreIndexを作成し、全DocumentをまとめてIndexing
   - StorageContextをpersist（永続化）

2. **追加インデックス作成時**
   - 既存のStorageContextをload
   - 新規会話セッションをDocumentに変換
   - 既存のVectorStoreIndexに追加insert
   - StorageContextを再persist

#### 3.2.3 Index構造
```
VectorStoreIndex {
  - nodes: [
      Node(会話1-chunk1, vector, metadata),
      Node(会話1-chunk2, vector, metadata),
      Node(会話2-chunk1, vector, metadata),
      ...
    ],
  - storage_context: StorageContext
}
```

### 3.3 データ更新・削除設計

#### 3.3.1 会話セッションの更新
- 既存会話セッションのuidで検索
- 該当するすべてのNodeを削除
- 更新後のDocumentから新しいNodeを生成・追加

#### 3.3.2 会話セッションの削除
- uidをmetadataで検索
- 該当するすべてのNode（チャンク）を削除

## 4. 検索設計

### 4.1 検索フロー
```
ユーザークエリ
  ↓
クエリのEmbedding化（qwen3-embedding:8b）
  ↓
ベクトル類似度検索（Milvus）
  ↓
Top-K候補取得（デフォルト: K=5）
  ↓
メタデータフィルタリング（オプション）
  ↓
関連チャンクの取得
  ↓
LLMによる回答生成（Qwen3-32B-AWQ）
  ↓
回答出力
```

### 4.2 検索モード

#### 4.2.1 基本検索
- **similarity_top_k**: 5
- 最も関連性の高いチャンクを取得

#### 4.2.2 メタデータフィルタ検索
特定条件で絞り込み検索
- 会社名指定検索
- 営業担当者名指定検索
- 期間指定検索
- 複合条件検索

#### 4.2.3 ハイブリッド検索
- ベクトル検索 + キーワード検索の組み合わせ
- Redis上のインデックスを活用した高速フィルタリング

### 4.3 Retriever設計

#### 4.3.1 VectorIndexRetriever
- 基本的なベクトル類似度検索
- similarity_top_kで取得数を制御

#### 4.3.2 カスタムRetriever
- メタデータフィルタ付き検索
- 時系列考慮検索（会話の前後関係を保持）
- 要約優先検索（要約フィールドから優先的に検索）

### 4.4 リランキング
取得した候補の再順位付け
- LLMベースのリランキング（クエリとの関連性を再評価）
- メタデータスコアリング（重要度に応じた重み付け）

## 5. クエリ処理・回答生成設計

### 5.1 QueryEngine構成
```
QueryEngine {
  - retriever: カスタムRetriever,
  - response_synthesizer: ResponseSynthesizer,
  - llm: Qwen3-32B-AWQ
}
```

### 5.2 回答生成戦略

#### 5.2.1 Compact（デフォルト）
- 取得した複数チャンクを1つのプロンプトにまとめてLLMに送信
- 最も効率的で一貫性のある回答

#### 5.2.2 Refine
- チャンクごとに段階的に回答を精緻化
- 長い文脈に有効

#### 5.2.3 TreeSummarize
- チャンクを階層的に要約しながら回答生成
- 大量の関連情報がある場合に有効

### 5.3 プロンプト設計

#### 5.3.1 システムプロンプト
```
あなたは営業支援AIアシスタントです。
営業と取引先の会話記録から、取引先のニーズや要望を正確に抽出して回答してください。

回答時の注意点：
- 会話記録に基づいた事実のみを回答
- 推測や憶測は避ける
- 具体的な会社名や担当者名も含めて回答
- 複数の会話に関連情報がある場合は統合して回答
```

#### 5.3.2 クエリプロンプトテンプレート
```
以下の会話記録から、{query}に関する情報を抽出してください。

会話記録：
{context}

回答形式：
- 取引先名：
- 担当者名：
- ニーズ・要望：
- 決定事項：
- 関連する会話の日時：
```

## 6. パフォーマンス最適化設計

### 6.1 インデックス最適化
- **Milvusインデックスタイプ**: IVF_FLAT または HNSW
  - IVF_FLAT: バランス型（検索速度と精度のバランス）
  - HNSW: 高速検索優先
- **パーティション戦略**: 会社名または日付でパーティション分割

### 6.2 キャッシング戦略
- **Redis活用**:
  - 頻繁に検索されるクエリの結果をキャッシュ
  - Embeddingのキャッシュ（同一クエリの再計算を回避）
  - メタデータ検索結果のキャッシュ

### 6.3 バッチ処理
- インデックス作成時は複数Document同時処理
- Embedding生成はバッチAPIを活用

## 7. データフロー詳細

### 7.1 インデックス作成時のデータフロー
```
既存DB
  ↓ (SQL/API)
会話セッションデータ取得
  ↓
Documentオブジェクト生成
  - テキスト: 発話内容を時系列結合
  - メタデータ: 会社名、担当者名、要約など
  ↓
SimpleNodeParser（チャンキング）
  - 512トークン/チャンク
  - 128トークンオーバーラップ
  ↓
各チャンクのEmbedding生成
  - qwen3-embedding:8b
  ↓
StorageContextへ保存
  - MilvusVectorStore: ベクトル保存
  - RedisIndexStore: インデックス情報保存
  - MongoDocumentStore: 元ドキュメント保存
  ↓
VectorStoreIndex構築完了
```

### 7.2 検索時のデータフロー
```
ユーザークエリ入力
  ↓
クエリのEmbedding生成
  - qwen3-embedding:8b
  ↓
Milvusでベクトル類似度検索
  - Top-K候補取得
  ↓
(オプション) メタデータフィルタリング
  - Redis上のインデックスで高速フィルタ
  ↓
MongoDBから元テキスト取得
  ↓
(オプション) リランキング
  ↓
QueryEngineで回答生成
  - 取得チャンク + クエリをLLMに送信
  - Qwen3-32B-AWQで回答生成
  ↓
回答出力
  - 回答テキスト
  - 参照元会話セッション情報
  - 信頼度スコア
```

## 8. ストレージ設計詳細

### 8.1 Milvus（ベクトルDB）
**Collection構造**
```
Collection: "conversation_embeddings"
  Fields:
    - id: int64 (auto-generated)
    - embedding: float_vector[8192]
    - node_id: varchar (LlamaIndexのNode ID)
  
  Index:
    - type: HNSW
    - metric: COSINE
    - params: {M: 16, efConstruction: 256}
```

**パーティション設計**
- 会社名ごとにパーティション分割（オプション）
- 日付範囲ごとにパーティション分割（オプション）

### 8.2 Redis（インデックスDB）
**データ構造**
```
Key-Value構造:
  - index_struct:{index_id} → インデックスメタデータ
  - node_mapping:{node_id} → Node情報
  - metadata_index:company:{会社名} → Set[node_id]
  - metadata_index:salesperson:{営業担当者名} → Set[node_id]
  - metadata_index:date:{YYYY-MM-DD} → Set[node_id]
```

### 8.3 MongoDB（ドキュメントDB）
**Collection構造**
```
Collection: "conversation_documents"
  Document:
    - _id: ObjectId
    - node_id: string (LlamaIndexのNode ID)
    - text: string (チャンクテキスト)
    - metadata: {
        uid: string,
        会社名: string,
        営業担当者名: string,
        取引先担当者名: string,
        拠点名: string,
        部署名: string,
        要約_全体概要: string,
        要約_トピック別: string,
        要約_決定事項: string,
        会話開始時間: ISODate,
        会話終了時間: ISODate,
        chunk_index: int,
        total_chunks: int
      }
```

**インデックス設計**
```
Indexes:
  - node_id: unique
  - metadata.uid: 1
  - metadata.会社名: 1
  - metadata.営業担当者名: 1
  - metadata.会話開始時間: 1
  - compound: {metadata.会社名: 1, metadata.会話開始時間: -1}
```

## 9. 運用設計

### 9.1 データライフサイクル
- **データ保持期間**: 3年（ビジネス要件により調整）
- **アーカイブ**: 古いデータは別ストレージへ移行
- **削除**: 保持期間経過後は完全削除

### 9.2 モニタリング項目
- インデックス作成速度（Document/秒）
- 検索レスポンス時間
- Embedding生成時間
- LLM応答時間
- ストレージ使用量
- 検索精度（ユーザーフィードバックベース）

### 9.3 メンテナンス
- **定期的なインデックス最適化**
  - Milvusのcompact実行
  - Redisのメモリ最適化
- **データ整合性チェック**
  - 3つのストレージ間の整合性確認
  - 孤立したNode削除

## 10. 拡張性設計

### 10.1 スケーラビリティ
- **水平スケーリング**
  - Milvus: Cluster構成でスケールアウト
  - Redis: Cluster/Sentinel構成
  - MongoDB: ReplicaSet/Sharding
- **垂直スケーリング**
  - Embedding/LLMサーバーのリソース増強

### 10.2 将来的な拡張
- マルチモーダル対応（音声、画像の直接処理）
- リアルタイムインデックス更新
- 感情分析の追加
- 話者識別情報の統合
- グラフDB連携（関係性分析）

