import requests
import re
from typing import List
from llama_index.core.schema import Document
from models import ConversationSession, ConversationSummary, Topic, Utterance
import sys
print(sys.path)
from ..infrastructure.relational_stores.query_executor import QueryExecutor

class DocumentLoader:
    def __init__(self, query_executor: QueryExecutor):
        self.query_executor = query_executor
        self.base_url = "http://192.168.100.19/transcription_tasks/"

    def load_data(self) -> List[ConversationSession]:
        df = self.query_executor.read_query("SELECT uid, tantousha, acc_name, site_name, dept_name, contact_name, file_creation_date FROM [seizo].[dbo].[v_文字起こし音声]")
        ret: List[ConversationSession] = []
        for row in df.itertuples():
            try:
                uid = row.uid
                url = self.base_url + uid
                response = requests.get(url)
                response.raise_for_status()
                json_data = response.json()
                json_summary_text = json_data.get("summary", "")
                json_utterances = json_data.get("utterances", [])
                
                # 発話情報をパース
                utterances = []
                for json_utterance in json_utterances:
                    u = Utterance(
                        start_time=json_utterance.get("start_time", 0.0),
                        end_time=json_utterance.get("end_time", 0.0),
                        content=json_utterance.get("text", "")
                    )
                    utterances.append(u)
                
                # 要約情報を抽出
                overall_summary = self._extract_overall_summary(json_summary_text)
                topics = self._extract_topics(json_summary_text)
                
                summary = ConversationSummary(
                    summary_text=overall_summary,
                    topics=topics
                )
                
                # ConversationSessionを作成
                session = ConversationSession(
                    uid=uid,
                    sales_person=row.tantousha,
                    company_name=row.acc_name,
                    office_name=row.site_name,
                    department=row.dept_name,
                    client_person=row.contact_name,
                    utterances=utterances,
                    summary=summary,
                    created_at=row.file_creation_date
                )
                ret.append(session)
                
            except requests.RequestException as e:
                print(f"Error fetching transcription for UID {uid}: {e}")
                continue
            except Exception as e:
                print(f"Error processing data for UID {uid}: {e}")
                continue

        return ret
    
    def _extract_topics(self, text: str) -> List[Topic]:
        """トピック抽出のロジックを実装 
        
        構成:
        ◻︎トピック別要約
        Topic1: <Topic1のタイトル>
        ・<内容>
        ・<内容>

        Topic2: <Topic2のタイトル>
        ・<内容>
        ・<内容>

        □...
        """

        if not text:
            return []

        # キャプチャ: '□トピック別要約' から次の '□' まで（またはテキスト末尾まで）
        pattern = r'□\s*トピック別要約\s*[:：]?\s*(.*?)(?=□|$)'
        m = re.search(pattern, text, re.S)
        if not m:
            return []

        section = m.group(1).strip()
        lines = section.splitlines()
        
        topics = []
        current_title = None
        current_summaries = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # トピックのタイトル行を検出 (例: 'Topic1: タイトル' or 'タイトル:')
            title_match = re.match(r'^(.+?)\s*[:：]\s*(.*)$', line)
            if title_match and not line.startswith('・'):
                # 前のトピックを保存
                if current_title is not None:
                    topics.append(Topic(title=current_title, summaries=current_summaries))
                
                # 新しいトピック開始
                current_title = title_match.group(1).strip()
                rest = title_match.group(2).strip()
                current_summaries = []
                
                # タイトル行の後に内容がある場合
                if rest:
                    if rest.startswith('・'):
                        current_summaries.append(rest.lstrip('・').strip())
                    else:
                        current_summaries.append(rest)
                continue

            # 箇条書き行
            if line.startswith('・'):
                item = line.lstrip('・').strip()
                if item:
                    current_summaries.append(item)
            else:
                # タイトルなしの内容行
                if current_title is None:
                    current_title = "その他"
                    current_summaries = []
                current_summaries.append(line)

        # 最後のトピックを保存
        if current_title is not None:
            topics.append(Topic(title=current_title, summaries=current_summaries))

        return topics
    
    def _extract_overall_summary(self, text: str) -> str:
        """ 全体概要抽出のロジックを実装 
        
        構成:
        □全体概要
        <内容>

        □トピック別要約
        """

        if not text:
            return ""

        # キャプチャ: '□全体概要' から次の '□' まで（またはテキスト末尾まで）
        pattern = r'□\s*全体概要\s*[:：]?\s*(.*?)(?=□|$)'
        m = re.search(pattern, text, re.S)
        if not m:
            return ""

        summary = m.group(1).strip()

        # 行頭の箇条書き記号を取り除き、余分な空行を削除
        lines = [line.lstrip('・').rstrip() for line in summary.splitlines()]
        # トリム
        while lines and lines[0] == '':
            lines.pop(0)
        while lines and lines[-1] == '':
            lines.pop()

        return "\n".join(lines).strip()