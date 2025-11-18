import logging
import sys
import os
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/rag_system/logs/rag_system.log")
    ]
)

logger = logging.getLogger(__name__)

from .experiment_runner import ExperimentRunner

def main():
    runner = ExperimentRunner(
        config_path="/config",
        data_dir="/sample_data/pdf",
        result_dir="/results"
    )
    runner.run_all_experiments()

if __name__ == "__main__":
    main()