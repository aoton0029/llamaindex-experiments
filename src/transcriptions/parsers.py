import logging
import re
from typing import List, Optional
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import NodeParser

from .models import ConversationSession, ConversationChunkMetadata, Utterance, ChunkType, Topic, ConversationSummary

logger = logging.getLogger(__name__)


class ConversationDocumentConverter:
    """ConversationSession → Document変換"""
    
    @staticmethod
    def session_to_documents(session: ConversationSession) -> List[Document]:
        """1セッションから複数のDocumentを生成（概要系と会話系）"""
        documents = []
        
        # 1. 全体概要Document
        summary_doc = ConversationDocumentConverter._create_summary_document(session)
        if summary_doc:
            documents.append(summary_doc)
        
        # 2. トピック別要約Documents
        topic_docs = ConversationDocumentConverter._create_topic_documents(session)
        documents.extend(topic_docs)
        
        # 3. 会話詳細Document
        conversation_doc = ConversationDocumentConverter._create_conversation_document(session)
        if conversation_doc:
            documents.append(conversation_doc)
        
        return documents
    
    @staticmethod
    def _create_summary_document(session: ConversationSession) -> Optional[Document]:
        """全体概要のDocument作成"""
        # session.summaryはConversationSummaryオブジェクト
        if not session.summary or not session.summary.summary_text:
            return None
        
        summary_text = session.summary.summary_text.strip()
        if not summary_text:
            return None
        
        metadata = {
            "session_uid": session.uid,
            "chunk_type": ChunkType.SUMMARY.value,
            "sales_person": session.sales_person,
            "company_name": session.company_name,
            "branch_name": session.branch_name or "",
            "department_name": session.department_name or "",
            "client_person": session.client_person,
        }
        
        return Document(
            text=summary_text,
            metadata=metadata,
            id_=f"{session.uid}_summary"
        )
    
    @staticmethod
    def _create_topic_documents(session: ConversationSession) -> List[Document]:
        """トピック別要約のDocuments作成"""
        documents = []
        
        # session.summary.topicsはList[Topic]
        if not session.summary or not session.summary.topics:
            return documents
        
        for topic in session.summary.topics:
            if not topic.title:
                continue
            
            # topic.contentsはList[str]
            topic_content = "\n".join([f"・{content}" for content in topic.contents])
            
            metadata = {
                "session_uid": session.uid,
                "chunk_type": ChunkType.TOPIC.value,
                "sales_person": session.sales_person,
                "company_name": session.company_name,
                "branch_name": session.branch_name or "",
                "department_name": session.department_name or "",
                "client_person": session.client_person,
                "topic_title": topic.title,
            }
            
            # トピックタイトル + 内容を結合
            full_text = f"{topic.title}\n{topic_content}"
            
            doc = Document(
                text=full_text,
                metadata=metadata,
                id_=f"{session.uid}_topic_{topic.title[:20]}"
            )
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def _create_conversation_document(session: ConversationSession) -> Optional[Document]:
        """会話詳細のDocument作成（発話を結合）"""
        if not session.utterances:
            return None
        
        # 全発話を時系列で結合
        conversation_lines = []
        for utt in session.utterances:
            time_str = f"[{utt.start_time:.1f}s-{utt.end_time:.1f}s]"
            conversation_lines.append(f"{time_str} {utt.content}")
        
        conversation_text = "\n".join(conversation_lines)
        
        metadata = {
            "session_uid": session.uid,
            "chunk_type": ChunkType.CONVERSATION.value,
            "sales_person": session.sales_person,
            "company_name": session.company_name,
            "branch_name": session.branch_name or "",
            "department_name": session.department_name or "",
            "client_person": session.client_person,
            "start_time": session.utterances[0].start_time,
            "end_time": session.utterances[-1].end_time,
        }
        
        return Document(
            text=conversation_text,
            metadata=metadata,
        )


class ConversationNodeParser(NodeParser):
    """会話データ用のカスタムNodeParser"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        time_window: Optional[float] = None,  # 秒単位の時間窓
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.time_window = time_window
    
    def _parse_nodes(
        self,
        nodes: List[TextNode],
        show_progress: bool = False,
        **kwargs,
    ) -> List[TextNode]:
        """Nodeのパース処理"""
        all_nodes = []
        
        for node in nodes:
            chunk_type = node.metadata.get("chunk_type")
            
            if chunk_type == ChunkType.CONVERSATION.value:
                # 会話は時間窓または文字数でチャンク化
                parsed = self._parse_conversation_node(node)
            else:
                # 概要・トピックはそのまま（すでに意味的な単位）
                parsed = [node]
            
            all_nodes.extend(parsed)
        
        return all_nodes
    
    def _parse_conversation_node(self, node: TextNode) -> List[TextNode]:
        """会話Nodeを時間窓またはチャンクサイズで分割"""
        # シンプルな実装: 文字数ベースで分割
        text = node.text
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # 時間情報を抽出（簡易版）
            time_match = re.search(r'\[(\d+\.?\d*)s-(\d+\.?\d*)s\]', chunk_text)
            chunk_start_time = float(time_match.group(1)) if time_match else None
            
            # 最後の時間を探す
            last_time_match = None
            for match in re.finditer(r'\[(\d+\.?\d*)s-(\d+\.?\d*)s\]', chunk_text):
                last_time_match = match
            chunk_end_time = float(last_time_match.group(2)) if last_time_match else None
            
            chunk_metadata = node.metadata.copy()
            if chunk_start_time is not None:
                chunk_metadata["start_time"] = chunk_start_time
            if chunk_end_time is not None:
                chunk_metadata["end_time"] = chunk_end_time
            
            chunk_node = TextNode(
                text=chunk_text,
                metadata=chunk_metadata,
                id_=f"{node.id_}_chunk_{len(chunks)}"
            )
            chunks.append(chunk_node)
            
            start = end - self.chunk_overlap
        
        return chunks if chunks else [node]

