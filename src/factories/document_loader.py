import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from llama_index.core.schema import Document
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.readers.file import (
    PDFReader, 
    DocxReader, 
    EpubReader, 
    MarkdownReader,
    PandasExcelReader
)
import pymupdf4llm
from pymupdf4llm.llama.pdf_markdown_reader import PDFMarkdownReader

logger = logging.getLogger(__name__)


class DocumentLoader():
    def __init__(self):
        self.loaders = {
            ".pdf": PDFReader(),
            ".docx": DocxReader(),
            ".epub": EpubReader(),
            ".md": MarkdownReader(),
            ".xlsx": PandasExcelReader(),
            ".xls": PandasExcelReader(),
        }

    def load_from_file(self, file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lower()
        if ext in self.loaders:
            loader = self.loaders[ext]
            documents = loader.load_data(file=file_path)
            logger.info(f"{file_path}から{len(documents)}ドキュメントを読み込みました")
            return documents
        else:
            raise ValueError(f"対応していないファイル形式: {ext}")
    
    def load_from_directory(self, dir_path: str) -> List[List[Document]]:
        path = Path(dir_path)
        all_documents = []
        for file_path in path.glob("**/*"):
            if file_path.is_file():
                documents = self.load_from_file(file_path)
                all_documents.append(documents)
        return all_documents
    
    def load_pdf_markdown(self, file_path: str) -> List[Document]:
        try:
            return PDFMarkdownReader().load_data(file=file_path)
        except Exception as e:
            logger.error(f"PDFMarkdownReaderエラー: {e}")
            return []
    
    def load_with_pymupdf4llm(self, file_path: str) -> str:
        return pymupdf4llm.to_markdown(file_path)
