from abc import ABC, abstractmethod
from typing import Optional, Union
from pathlib import Path
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from markdownify import markdownify as md
import re
import time
from urllib.parse import urljoin
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImportBase(ABC):
    def __init__(self, source_url):
        self.source_url = source_url

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement run method") 

    @abstractmethod
    def parse_data(self, raw_data: str):
        raise NotImplementedError("Subclasses must implement parse_data method")

    def save_data(self, out_path, parsed_data: str):
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(parsed_data, encoding="utf-8")
        logger.info("Saved markdown to %s", out_file)

    def import_data(self, url, out_path, timeout=10, headers=None, retries=2):
        raw_data = self.fetch_data(url, timeout, headers, retries)
        parsed_data = self.parse_data(raw_data)
        self.save_data(out_path, parsed_data)

    def fetch_data(self,
        url: str,
        timeout: int = 10,
        headers: Optional[dict] = None,
        retries: int = 2,
    ) -> str:
        """
        URLからHTMLを取得してmarkdownifyでMarkdownに変換する。
        out_pathを指定するとファイルに保存する。戻り値はMarkdown文字列。
        """
        session = requests.Session()
        retry = Retry(total=retries, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))

        headers = headers or {"User-Agent": "python-requests/urllib3"}
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch URL %s: %s", url, e)
            raise

        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        html = resp.content.decode(encoding, errors="replace")

        md_text = md(html, heading_style="ATX")
        return md_text


class GlossaryHomeImporter(ImportBase):
    BASE_URL = "https://www.matsusada.co.jp/support/terms/"
    def __init__(self):
        super().__init__(self.BASE_URL)

    def parse_data(self, raw_data: str) -> str:
        """
        - 「# 用語集」と一致する行より上の行を全削除（その行も削除する）
        """
        pattern = r"^# 用語集.*$"
        match = re.search(pattern, raw_data, flags=re.MULTILINE)
        if match:
            raw_data = raw_data[match.end():]
        return raw_data

    def run(self):
        out_file = Path(".") / "glossary_home.md"
        self.import_data(self.source_url, out_path=out_file)



class GlossaryTermImporter(ImportBase):
    BASE_URL = "https://www.matsusada.co.jp"
    def __init__(self):
        super().__init__(self.BASE_URL)
    
    def parse_data(self, raw_data: str) ->  str:
        """
        glossary_dir 内の全ての .md ファイルに対して次の編集を行う。
        - 「用語集」と一致する行以上の全行を削除
        - 「## [用語集]」を含む行以下の全行を削除
        - 改行のみの行（空行）を削除する
        """
        pattern_start = r"^用語集.*$"
        match_start = re.search(pattern_start, raw_data, flags=re.MULTILINE)
        if match_start:
            raw_data = raw_data[match_start.end():]

        pattern_end = r"^## \[用語集\].*$"
        match_end = re.search(pattern_end, raw_data, flags=re.MULTILINE)
        if match_end:
            raw_data = raw_data[:match_end.start()]

        # 空行を削除
        lines = raw_data.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        return "\n".join(non_empty_lines)
        
    def run(self):
        home_file_path = Path(".") / "glossary_home.md"
        home_content = home_file_path.read_text(encoding="utf-8")
        links = re.findall(r"\[([^\]]+)\]\((/support/terms/[^)]+)\)", home_content)
        for title, link in links:
            url = urljoin(self.source_url, link)
            safe_title = re.sub(r"[\\/*?\"<>|]", "_", title)
            out_file = Path(".") / "datas" / "glossary" / f"{safe_title}.md"
            try:
                self.import_data(url, out_path=out_file)
                time.sleep(3)  # Be polite with server
                break
            except Exception as e:
                logger.error("Error processing %s: %s", url, e)



class TechColumnImporter(ImportBase):
    BASE_URL = "https://www.matsusada.co.jp/column/"
    def __init__(self):
        super().__init__(self.BASE_URL)

    def parse_data(self, raw_data: str) -> str:
        """
        - 「# 技術コラム」と一致する行より上の行を全削除
        - 「1」と一致する行または「[前の10件](/column/index.html)」と一致する行以下の全行を削除
        """
        pattern = r"^# 技術コラム.*$"
        match = re.search(pattern, raw_data, flags=re.MULTILINE)
        if match:
            raw_data = raw_data[match.end():]
        
        pattern_end = r"^(1|\[前の10件\]\(/column/index.html\)).*$"
        match_end = re.search(pattern_end, raw_data, flags=re.MULTILINE)
        if match_end:
            raw_data = raw_data[:match_end.start()]
        
        return raw_data
        

    def run(self):
        for i in range(1, 12):
            url = urljoin(self.source_url, f"index_{i}.html" if i > 1 else "")
            out_file = Path(".") / "datas" / "tech_column" / f"index_{i}.md"
            try:
                self.import_data(url, out_path=out_file)
                time.sleep(3)  # Be polite with server
            except Exception as e:
                logger.error("Error processing %s: %s", url, e)


if __name__ == "__main__":
    glossary_home_importer = GlossaryHomeImporter()
    glossary_home_importer.run()

    glossary_importer = GlossaryTermImporter()
    glossary_importer.run()

    tech_column_importer = TechColumnImporter()
    tech_column_importer.run()