"""
Main Execution Script / CLI Entrypoint.
Includes Latency, Memory Usage, Retrieval Scoring (Recall@K, MRR), and Evaluation Metrics (ROUGE, BLEU, BERTScore).
"""

# ------------------------------------
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from evaluation import RAGComparisonPipeline
import pandas as pd
# ... rest of your code
import os
import argparse
from config import LLM_CONFIGURATIONS, LLM_MOCK_CONFIGURATIONS
from data_loader import load_project_dataset
from evaluation import RAGComparisonPipeline


def main():
    parser = argparse.ArgumentParser(description="Modular RAG + LLM Comparison Framework")
    parser.add_argument("--data_path", type=str, default="processed_news.csv", help="Path to CSV dataset")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to index")
    parser.add_argument("--query", type=str, default="US president discusses new economic policy", help="Target test query")
    parser.add_argument("--output_file", type=str, default="rag_llm_results.csv", help="Output file path (.csv or .xlsx)")
    parser.add_argument("--mock", action="store_true", help="Run with mock LLM outputs for rapid testing")
    args = parser.parse_args()

    llm_configs = LLM_MOCK_CONFIGURATIONS if args.mock else LLM_CONFIGURATIONS


    print("--- 1. Loading Dataset ---")
    data_file = args.data_path if os.path.exists(args.data_path) else "processed_news.csv"
    df = load_project_dataset(local_path=data_file, n_samples=args.n_samples)
    print(f"Loaded {len(df)} records.")

    # FIX: Dynamically find reference summary column
    summary_col = "Summary" if "Summary" in df.columns else ("highlights" if "highlights" in df.columns else df.columns[1])
    
    # Select sample 0 as target query context
    target_idx = 0
    ref_summary = df[summary_col].iloc[target_idx] if not df.empty else ""
    
    # Provide a meaningful query based on article context if using default query
    test_query = args.query if args.query != "US president discusses new economic policy" else df["Article"].iloc[target_idx][:100]

    print("\n--- 2. Building Index & Executing Matrix (3 RAG x 3 LLMs) ---")
    pipeline = RAGComparisonPipeline(df=df, llm_configs=llm_configs)
    
    # FIX: Pass target_doc_id so Recall@K and MRR don't return 0
    results_df = pipeline.run_all_combinations(query=test_query, reference_summary=ref_summary, target_doc_id=target_idx)

    print("\n=== EXECUTION SUMMARY ===")
    display_cols = [
        "rag_technique", "llm_model", "total_latency_sec", 
        "memory_mb", "recall_at_k", "mrr", "rouge1", "bleu", "bert_score"
    ]
    print(results_df[display_cols].to_string(index=False))

    if args.output_file.endswith(".xlsx"):
        results_df.to_excel(args.output_file, index=False)
    else:
        results_df.to_csv(args.output_file, index=False)
        
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
    print(results_df[['rag_technique', 'llm_model', 'generated_text']])