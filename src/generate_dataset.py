import logging
import sys
import os
print(os.environ["TIKTOKEN_CACHE_DIR"])
import json
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from llama_index.core import Settings, StorageContext
from llama_index.core.indices.base import BaseIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from db.database_manager import DatabaseConfig, DatabaseManager
from transformers import AutoTokenizer
from services import ConfigManager
from factories import (
    MyTestsetGenerator,
    LlamaIndexDatasetFactory,
    LLMFactory,
    EmbeddingFactory,
    DocumentLoader
)

logger = logging.getLogger(__name__)


llamadebughandler = LlamaDebugHandler()
callback_manager = CallbackManager([llamadebughandler])
Settings.callback_manager = callback_manager
Settings.llm = LLMFactory.create(backend="vllm", model_name="/models/Qwen/Qwen3-VL-30B-A3B-Instruct", base_url="http://vllm-llm:8000/v1")
Settings.embed_model = EmbeddingFactory.create(backend="vllm", model_name="/models/Qwen/Qwen3-VL-8B-Instruct", base_url="http://vllm-embedding:8000/v1")
Settings.tokenizer = AutoTokenizer.from_pretrained("/workspace/tokenizer_data/cl-nagoya-ruri-v3-310m")

def generate_dataset(source_type: str, source_path: str, output_path: str, num_questions: int = 3) -> None:
    framework = "llamaindex"  # "llamaindex" または "ragas" を選択
    loader = DocumentLoader()
    documents = loader.simple_load_from_directory(source_path)
    if framework == "llamaindex":
        dataset_factory = LlamaIndexDatasetFactory()
        dataset = dataset_factory.create_dataset_from_documents(
            documents=documents,
            num_questions=num_questions,
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dataset_factory.save_dataset_to_json(dataset, output_path)

def main():
    try:
        generate_dataset(
            source_type="directory",
            source_path="/workspace/sample_data/pdf",
            output_path="/workspace/test6/results/sample_ragas_dataset.json",
            num_questions=3
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
