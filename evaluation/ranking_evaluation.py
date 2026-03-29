import argparse
import csv
import json
import os
from statistics import mean

from datasets import load_dataset
from tabulate import tabulate

from blagent.util.code_util import extract_patch_file_path

K_VALUES = [1, 3, 5, 10]

swebench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")


def evaluate_retrieval(patch_file_path, retrieved_files, top_k=K_VALUES):
    result = {}
    try:
        rank = retrieved_files.index(patch_file_path) + 1
        result["rank"] = rank
        result["reciprocal_rank"] = 1 / rank
    except ValueError:
        result["rank"] = None
        result["reciprocal_rank"] = 0.0

    for k in top_k:
        in_top_k = patch_file_path in retrieved_files[:k]
        result[f"top_{k}"] = in_top_k

    return result


def compute_aggregate_stats(all_metrics, k_values=K_VALUES):
    stats = {
        "MRR": (
            mean(m.get("reciprocal_rank", 0.0) for m in all_metrics)
            if all_metrics
            else 0.0
        ),
    }

    for k in k_values:
        stats[f"Acc@{k}"] = (
            mean(m.get(f"top_{k}", 0.0) for m in all_metrics) if all_metrics else 0.0
        )

    return stats


def normalize_ranked_files(pred, pred_list_name):
    if pred_list_name == "ranked_scores":
        ranked_files = pred.get("ranked_scores", {})
        return [file for file, _ in sorted(ranked_files.items(), key=lambda x: -x[1])]

    if pred_list_name == "final_reranked_files":
        final_files = pred.get("final_reranked_files", [])
        if final_files and isinstance(final_files[0], dict):
            return [file for d in final_files for file in d.keys()]
        return final_files

    return pred.get(pred_list_name, [])


def evaluate_predictions(
    predictions, pred_list_name, ground_truth, k_values=K_VALUES, limit=None
):
    all_metrics = []

    if limit is not None:
        predictions = predictions[:limit]

    failed_cases = 0
    cases_where_retrieval_failed = 0

    for i, pred in enumerate(predictions):
        if ground_truth == "patch":
            patch_file_path = extract_patch_file_path(pred.get("patch", ""))
        elif ground_truth == "patch_file":
            patch_file_path = pred.get("patch_file", "")
        else:
            raise ValueError(f"Unknown ground_truth type: {ground_truth}")

        if not patch_file_path:
            print(f"[Warning] Could not extract patch file in item {i}")
            continue

        ranked_files = normalize_ranked_files(pred, pred_list_name)

        if patch_file_path not in ranked_files[:10]:
            failed_cases += 1
            t0_files = pred.get("retrieved_files_t0", [])
            t1_files = pred.get("retrieved_files_t1", [])
            if (
                patch_file_path not in t0_files[:15]
                and patch_file_path not in t1_files[:15]
            ):
                cases_where_retrieval_failed += 1

        metrics = evaluate_retrieval(patch_file_path, ranked_files, top_k=k_values)
        all_metrics.append(metrics)

    stats = compute_aggregate_stats(all_metrics, k_values)
    stats["RF / Top10 Misses"] = (
        f"{cases_where_retrieval_failed}/{failed_cases}" if failed_cases else "0/0"
    )
    return stats


def stats_to_row(report, stats, k_values=K_VALUES):
    row = {
        "RQ": report["rq"],
        "Setting": report["name"],
        "MRR": round(stats["MRR"], 4),
    }
    for k in k_values:
        row[f"Acc@{k}"] = round(stats[f"Acc@{k}"], 4)
    row["RF / Top10 Misses"] = stats["RF / Top10 Misses"]
    return row


def print_master_table(rows, k_values=K_VALUES):
    headers = (
        ["RQ", "Setting", "MRR"]
        + [f"Acc@{k}" for k in k_values]
        + ["RF / Top10 Misses"]
    )
    table = [[row[h] for h in headers] for row in rows]

    print("\n==================== ALL RESULTS ====================")
    print(tabulate(table, headers=headers, tablefmt="github"))


def export_csv(rows, output_csv, k_values=K_VALUES):
    headers = (
        ["RQ", "Setting", "MRR"]
        + [f"Acc@{k}" for k in k_values]
        + ["RF / Top10 Misses"]
    )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Saved CSV] {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ranking predictions and generate one consolidated table."
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/reviewer_tables/all_rq_results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    reports = [
        # RQ 1-1
        {
            "rq": "RQ1.1",
            "name": "BLAgent (GPT-OSS 120B)",
            "path": "results/rq1-1/blagent_gpt-oss.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.1",
            "name": "BLAgent (Claude 4.6 Sonnet)",
            "path": "results/rq1-1/blagent_claude-4-6.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        # RQ 1-2
        {
            "rq": "RQ1.2",
            "name": "RAG - Text Chunking",
            "path": "results/rag_results_text_splitter.json",
            "pred_list_name": "rag_ranked_files",
            "ground_truth": "patch",
        },
        {
            "rq": "RQ1.2",
            "name": "RAG - Code Aware Chunking",
            "path": "results/rq1-2/rag_code_chunking.json",
            "pred_list_name": "rag_ranked_files",
            "ground_truth": "patch",
        },
        {
            "rq": "RQ1.2",
            "name": "RAG - File Path Aware Code Chunking",
            "path": "results/rq1-2/rag_file_path_augmented_code_chunking.json",
            "pred_list_name": "rag_ranked_files",
            "ground_truth": "patch",
        },
        # RQ 1-3
        {
            "rq": "RQ1.3",
            "name": "BLAgent (GPT-OSS 120B) - Base Retrieval (Phase 1)",
            "path": "results/rq1-3/blagent_gpt-oss_phase1_and_phase2_base_retrieval.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.3",
            "name": "BLAgent (GPT-OSS 120B) - Base Retrieval (Phase 1 + Phase 2)",
            "path": "results/rq1-3/blagent_gpt-oss_phase1_and_phase2_base_retrieval.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        # RQ 1-4
        {
            "rq": "RQ1.4",
            "name": "BLAgent (Phase 1) T0 (GPT-OSS 120B)",
            "path": "results/rq1-4/agentic_t0_retrieval_gpt-oss_120b_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.4",
            "name": "BLAgent (Phase 1) T1 (GPT-OSS 120B)",
            "path": "results/rq1-4/agentic_t1_retrieval_gpt-oss_120b_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.4",
            "name": "BLAgent (Phase 1) T0 + T1 (GPT-OSS 120B) - T0 First",
            "path": "results/rq1-4/agentic_t0_t1_retrieval_gpt-oss_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.4",
            "name": "BLAgent (Phase 1) T0 + T1 (GPT-OSS 120B) - T1 First",
            "path": "results/rq1-4/agentic_t0_t1_retrieval_gpt-oss_ranked_results_t1_first.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        # RQ 1-5
        {
            "rq": "RQ1.5",
            "name": "BLAgent (Claude 4.6) - Phase 1",
            "path": "results/rq1-5/different_llms/phase1_claude_4_6_sonnet_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.5",
            "name": "BLAgent (Claude 4.6) - Phase 1 + Phase 2",
            "path": "results/rq1-5/different_llms/phase2_claude_4_6_sonnet_ranked_results.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.5",
            "name": "BLAgent (Qwen3 32B) - Phase 1",
            "path": "results/rq1-5/different_llms/phase1_qwen3_32b_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.5",
            "name": "BLAgent (Qwen3 32B) - Phase 1 + Phase 2",
            "path": "results/rq1-5/different_llms/phase2_qwen3_32b_ranked_results.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.5",
            "name": "BLAgent (GPT-OSS 120B) - Phase 1",
            "path": "results/rq1-5/different_llms/phase1_gpt-oss_120b_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "rq": "RQ1.5",
            "name": "BLAgent (GPT-OSS 120B) - Phase 1 + Phase 2",
            "path": "results/rq1-5/different_llms/phase2_gpt-oss_120b_ranked_results.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        # Discussion 5-5
        {
            "rq": "Dis 5.5",
            "name": "Pruned Context 5 files 5 chunks (GPT-OSS)",
            "path": "results/chunk_analysis/phase2_5files_5chunk.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        {
            "rq": "Dis 5.5",
            "name": "Pruned Context 5 files 10 chunks (GPT-OSS)",
            "path": "results/chunk_analysis/phase2_5files_10chunk.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        {
            "rq": "Dis 5.5",
            "name": "Pruned Context 10 files 5 chunks (GPT-OSS)",
            "path": "results/chunk_analysis/phase2_10files_5chunk.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
        {
            "rq": "Dis 5.5",
            "name": "Pruned Context 10 files 10 chunks (GPT-OSS)",
            "path": "results/chunk_analysis/phase2_10files_10chunk.json",
            "pred_list_name": "final_reranked_files",
            "ground_truth": "patch_file",
        },
    ]

    all_rows = []

    for report in reports:
        print(f"Evaluating: {report['rq']} | {report['name']}")
        with open(report["path"], "r") as f:
            predictions = json.load(f)

        stats = evaluate_predictions(
            predictions=predictions,
            pred_list_name=report["pred_list_name"],
            ground_truth=report["ground_truth"],
            limit=args.limit,
        )

        all_rows.append(stats_to_row(report, stats))

    print_master_table(all_rows)
    # export_csv(all_rows, args.csv)
