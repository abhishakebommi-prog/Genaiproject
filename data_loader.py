"""
Data loader module for project datasets with fallback to Hugging Face cnn_dailymail.
"""

import os
import pandas as pd
from datasets import load_dataset


def load_project_dataset(local_path: str = "processed_news.csv", n_samples: int = 5000) -> pd.DataFrame:
    """
    Loads dataset from local CSV if valid and non-empty. 
    Otherwise, downloads cnn_dailymail from Hugging Face, saves it to local CSV, and returns it.
    """
    # Check if local CSV exists AND is non-empty (>0 bytes)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        try:
            df = pd.read_csv(local_path, encoding="utf-8")
            if not df.empty:
                print(f"Loaded existing dataset from {local_path} ({len(df)} rows).")
                return df.head(n_samples)
        except Exception as e:
            print(f"Warning: Could not read {local_path} ({e}). Downloading fresh dataset...")

    # If missing, empty, or corrupted -> load from Hugging Face
    print(f"'{local_path}' not found or empty. Loading cnn_dailymail/3.0.0 from Hugging Face...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{n_samples}]")
    
    df = pd.DataFrame({
        "Article": ds["article"],
        "Summary": ds["highlights"]
    })

    # Save clean dataset locally so future runs load instantly
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    df.to_csv(local_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} records to {local_path}.")

    return df


# Protect top-level execution when imported by main.py
if __name__ == "__main__":
    news_df = load_project_dataset()
    print("Dataset confirmed.")
    print(news_df.head(2))