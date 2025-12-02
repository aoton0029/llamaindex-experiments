# エクストラクタパターン定義
extractor_patterns:
  # タイトル + サマリー + キーワード
  pattern_title_summary_keyword:
    name: "タイトル + サマリー + キーワード"
    extractors:
      - type: "title"
        kwargs:
          nodes: 5
      - type: "summary"
        kwargs: {}
      - type: "keyword"
        kwargs:
          keywords: 5
  
  # フルエクストラクタ
  pattern_full:
    name: "フルエクストラクタ"
    extractors:
      - type: "title"
        kwargs:
          nodes: 5
      - type: "summary"
        kwargs: {}
      - type: "keyword"
        kwargs:
          keywords: 5
      - type: "questions_answered"
        kwargs:
          questions: 5
  
  # タイトルのみ
  pattern_title_only:
    name: "タイトルのみ"
    extractors:
      - type: "title"
        kwargs:
          nodes: 5
  
  # タイトル + キーワード
  pattern_title_keyword:
    name: "タイトル + キーワード"
    extractors:
      - type: "title"
        kwargs:
          nodes: 5
      - type: "keyword"
        kwargs:
          keywords: 5
  
  # サマリー + キーワード
  pattern_summary_keyword:
    name: "サマリー + キーワード"
    extractors:
      - type: "summary"
        kwargs: {}
      - type: "keyword"
        kwargs:
          keywords: 5
  
  # タイトル + サマリー
  pattern_title_summary:
    name: "タイトル + サマリー"
    extractors:
      - type: "title"
        kwargs:
          nodes: 5
      - type: "summary"
        kwargs: {}
  
  # 質問重視
  pattern_questions_focus:
    name: "質問重視"
    extractors:
      - type: "title"
        kwargs:
          nodes: 5
      - type: "questions_answered"
        kwargs:
          questions: 10
