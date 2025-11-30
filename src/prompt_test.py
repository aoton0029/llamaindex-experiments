import os
import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from llama_index.core.llms.llm import BaseLLM
from llama_index.core.prompts import PromptTemplate
from factories import (
    LLMFactory,
    ExtractorFactory,
    DocumentLoader
)
from services.config_manager import ConfigManager
from factories.template_prompts import TemplatePromptSettings
from llama_index.core import Settings

from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
llamadebughandler = LlamaDebugHandler()
callback_manager = CallbackManager([llamadebughandler])
Settings.callback_manager = callback_manager

# from llama_index.core import set_global_handler
# set_global_handler("langfuse")

logger = logging.getLogger(__name__)

def test_template(llm: BaseLLM, template: PromptTemplate, template_name: str, **kwargs):
    """
    指定されたテンプレートをLLMに送信してテストする
    """
    print(f"\n{'='*80}")
    print(f"テスト: {template_name}")
    print(f"{'='*80}")
    
    try:
        # テンプレートをフォーマット
        prompt = template.format(**kwargs)
        print(f"【送信プロンプト】{len(prompt)}文字\n{prompt[:300]}")
        print("-"*80)
        
        response = llm.complete(prompt)
        
        print(f"【LLMレスポンス】")
        print(f"{response.text}")
        print("-"*80)
        print(f"✓ テスト成功")
        
        return True
        
    except Exception as e:
        print(f"\n✗ テスト失敗: {str(e)}")
        return False

def test_title(llm, context_str: str):
    return test_template(
        llm,
        TemplatePromptSettings.JP_TITLE_NODE_TEMPLATE,
        "Title",
        context_str=context_str
    )
    
def test_summary(llm, context_str: str):
    return test_template(
        llm,
        TemplatePromptSettings.JP_SUMMARY_EXTRACT_TEMPLATE,
        "Summary",
        context_str=context_str
    )

def test_keywords(llm, context_str: str):
    return test_template(
        llm,
        TemplatePromptSettings.JP_KEYWORD_EXTRACT_TEMPLATE,
        "Keywords",
        text=context_str,
        max_keywords = 5
    )

def main():
    """メイン関数"""
    config_manager = ConfigManager("/workspace/src/config")
    TemplatePromptSettings.initialize(config_manager.get_template_prompts())
    # LLM設定（環境に応じて変更してください）
    backend = "vllm" 
    model_name = "/models/Llama-3-ELYZA-JP-8B-AWQ"
    base_url = "http://vllm:8000/v1"
    
    print("LLMの初期化中...")
    try:
        llm = LLMFactory.create(
            backend=backend,
            model_name=model_name,
            base_url=base_url,
            temperature=0.0,
            timeout=180.0,
            max_tokens=2048,
        )
        Settings.llm = llm
        print("✓ LLM初期化完了")
    except Exception as e:
        print(f"✗ LLM初期化失敗: {str(e)}")
        return
    
    loader = DocumentLoader()
    # テストデータ
    docs = loader.load_from_file("/workspace/datas/tech_column/terms/「高圧電気取扱特別教育」って何？.md")

    test_results = []    
    for doc in docs:
        context_str = doc.text
        test_results.append(test_title(llm, context_str))
        # test_results.append(test_summary(llm, context_str))
        # test_results.append(test_keywords(llm, context_str))

    # 結果サマリー
    print(f"\n{'='*80}")
    print("テスト結果サマリー")
    print(f"{'='*80}")
    print(f"成功: {sum(test_results)}/{len(test_results)}")
    print(f"失敗: {len(test_results) - sum(test_results)}/{len(test_results)}")


if __name__ == "__main__":
    main()
