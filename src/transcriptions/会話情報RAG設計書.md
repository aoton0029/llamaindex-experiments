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

#### 3.1.2 既存DBスキーマ
##### 3.1.2.1 会話セッション
- uid(会話セッションのユニークID)
- 営業担当者名
- 会社名
- 拠点名
- 部署名
- 取引先担当者名
- 会話情報(発話情報のリスト)
- 概要

##### 3.1.2.2 発話情報(話者不明)
- 発話開始時間(s)
- 発話終了時間(s)
- 発話内容

##### 3.1.2.3 会話セッションの要約の構成
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
```

#### 3.1.3 ドキュメント構造設計
各会話セッションを複数のDocumentに分割して扱う。

##### 3.1.3.1 概要インデックス用ドキュメント
**全体概要Document**
- chunk_type: "summary"
- テキスト: 全体概要の内容
- メタデータ: session_uid, 会社名, 営業担当者など

**トピック別Document (複数)**
- chunk_type: "topic"
- テキスト: トピックタイトル + トピック内容
- メタデータ: session_uid, topic_title, 会社名など

##### 3.1.3.2 会話詳細インデックス用ドキュメント
**会話詳細Document**
- chunk_type: "conversation"
- テキスト: 発話を時系列で結合（時間情報付き）
- メタデータ: session_uid, start_time, end_time, 会社名など
- チャンキング: 512文字単位で分割（時間窓考慮可能）

### 3.2 インデックス戦略

#### 3.2.1 2つのVectorStoreIndexを使用
**同一StorageContext内で2つのインデックスを構築**

1. **概要インデックス** (summary_index)
   - 対象: 全体概要 + トピック別要約
   - 目的: 大まかな絞り込み（どの会社が何に興味があるか）
   - チャンク化: 不要（すでに意味的な単位）

2. **会話詳細インデックス** (conversation_index)
   - 対象: 発話内容
   - 目的: 具体的な発言内容の検索
   - チャンク化: 512文字単位（chunk_overlap=50）

**利点:**
- 概要と詳細で異なる検索戦略を適用可能
- メタデータフィルタを使った柔軟な検索
- 同一StorageContextで一貫性を保持

### 3.3 検索戦略設計

#### 3.3.1 レトリーバー選定

**HybridConversationRetriever (2段階検索)**
```
Stage 1: 概要インデックスから関連セッションを特定
         ↓
Stage 2: 特定されたセッションの会話詳細を検索
```

**ChunkTypeFilteredRetriever (種別別検索)**
- 概要のみ検索: ChunkType.SUMMARY + ChunkType.TOPIC
- 会話のみ検索: ChunkType.CONVERSATION
- メタデータフィルタと組み合わせ可能

#### 3.3.2 クエリエンジン選定

**1. ハイブリッドクエリエンジン** (推奨)
- 用途: 汎用的な質問
- 例: 「ABC商事はどのようなシステムを求めていますか？」
- 特徴: 概要で絞り込み → 会話詳細で精緻化

**2. 概要専用クエリエンジン**
- 用途: 大まかな質問
- 例: 「どの会社が配送最適化に興味がありますか？」
- 特徴: 高速、全体像把握に最適

**3. 会話専用クエリエンジン**
- 用途: 具体的な発言の検索
- 例: 「〇〇についてどのような発言がありましたか？」
- 特徴: 詳細な内容、時間情報付き

### 3.4 応用可能な拡張

#### 3.4.1 BM25との組み合わせ (キーワード検索強化)
```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# Vector + BM25のハイブリッド
bm25_retriever = BM25Retriever.from_defaults(
    nodes=all_nodes,
    similarity_top_k=5
)
fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rerank"
)
```

#### 3.4.2 RouterQueryEngineによる自動振り分け
```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# 質問内容に応じて自動的にエンジンを選択
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_engine_tool,
        conversation_engine_tool
    ]
)
```

#### 3.4.3 SubQuestionQueryEngineによる複雑な質問分解
```python
from llama_index.core.query_engine import SubQuestionQueryEngine

# 複雑な質問を自動分解して回答
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        summary_engine_tool,
        conversation_engine_tool
    ]
)
```

## 4. メタデータ設計

### 4.1 共通メタデータ
- session_uid (必須): 会話セッションの一意識別子
- chunk_type (必須): summary / topic / conversation
- sales_person: 営業担当者名
- company_name: 会社名
- branch_name: 拠点名
- department_name: 部署名
- client_person: 取引先担当者名

### 4.2 種別固有メタデータ
**トピック (chunk_type="topic")**
- topic_title: トピックタイトル

**会話 (chunk_type="conversation")**
- start_time: 開始時間 (秒)
- end_time: 終了時間 (秒)

## 5. 実装フロー

### 5.1 インデクシングフロー
```
1. ConversationSessionデータ取得
   ↓
2. ConversationDocumentConverter.session_to_documents()
   → 全体概要Document
   → トピックDocuments (複数)
   → 会話詳細Document
   ↓
3. chunk_typeでフィルタリング
   → summary_docs (概要 + トピック)
   → conversation_docs (会話)
   ↓
4. VectorStoreIndex作成 (同一StorageContext使用)
   → summary_index
   → conversation_index
```

### 5.2 検索フロー
```
ユーザークエリ
   ↓
[質問タイプ判定] (オプション: RouterQueryEngine)
   ↓
┌─────────┬──────────────┬─────────────┐
│ 汎用    │ 大まかな質問 │ 詳細な質問  │
│ Hybrid  │ Summary Only │ Conversation│
└─────────┴──────────────┴─────────────┘
   ↓            ↓              ↓
2段階検索    概要検索       会話検索
   ↓            ↓              ↓
LLMによる回答生成
   ↓
回答出力 (会社名・担当者名・時間情報付き)
```

## 6. 推奨設定

### 6.1 パラメータ推奨値
- **similarity_top_k**: 5 (概要検索)、10 (会話検索)
- **chunk_size**: 512 (会話チャンク)
- **chunk_overlap**: 50
- **response_mode**: COMPACT (精度重視の場合はREFINE)
- **enable_two_stage**: True (2段階検索推奨)

### 6.2 ベクトルDBインデックス設定
- **metric_type**: COSINE
- **index_type**: HNSW (高速検索)
- **M**: 16-32 (グラフ接続数)
- **efConstruction**: 256-512 (構築精度)
