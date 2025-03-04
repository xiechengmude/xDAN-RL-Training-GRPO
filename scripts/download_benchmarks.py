import os
import requests
from pathlib import Path
import time

def download_file(url, target_path):
    """Download a file with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {url} to {target_path}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retrying
    return False

def main():
    # Define the benchmark datasets to download
    benchmarks = [
        "tinyBenchmarks/tinyBenchmarks.pkl",
        "tinyBenchmarks/tinyWinogrande",
        "tinyBenchmarks/tinyAI2_arc",
        "tinyBenchmarks/tinyHellaswag",
        "tinyBenchmarks/tinyMMLU",
        "tinyBenchmarks/tinyTruthfulQA",
        "tinyBenchmarks/tinyGSM8k"
    ]

    # Create cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download each benchmark
    for benchmark in benchmarks:
        url = f"https://huggingface.co/datasets/{benchmark}/resolve/main/data/dataset.pkl"
        target_path = cache_dir / benchmark / "dataset.pkl"
        print(f"\nDownloading {benchmark}...")
        if download_file(url, str(target_path)):
            print(f"Successfully cached {benchmark}")
        else:
            print(f"Failed to download {benchmark}")

if __name__ == "__main__":
    main()
