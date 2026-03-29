import json
import re
from statistics import mean

from datasets import load_dataset

swebench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")


def extract_patch_file_path(patch_str):
    """Extract the file path that was patched from the diff."""
    match = re.search(r"^diff --git a/(.*?) b/\1", patch_str, re.MULTILINE)
    if match:
        return match.group(1)
    fallback = re.search(r"^\+\+\+ b/(.+)", patch_str, re.MULTILINE)
    if fallback:
        return fallback.group(1)
    return None


# Read jsonl file
def read_jsonl_file(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def evaluate_retrieval(patch_file_path, retrieved_files, top_k=(1, 3, 5, 10)):
    result = {}
    retrieved_files = list(dict.fromkeys(retrieved_files))  # Remove duplicates
    try:
        rank = retrieved_files.index(patch_file_path) + 1
        result["rank"] = rank
        result["reciprocal_rank"] = 1 / rank
    except ValueError:
        result["rank"] = None
        result["reciprocal_rank"] = 0.0

    for k in top_k:
        result[f"top_{k}"] = patch_file_path in retrieved_files[:k]

    return result


def evaluate_predictions(predictions):
    all_metrics = []
    for i, pred in enumerate(predictions):
        _id = pred["instance_id"]
        patch = [d for d in swebench if d["instance_id"] == _id][0]["patch"]
        patch_file_path = patch  # pred.get("patch", "")
        retrieved_files = pred.get("found_files", [])
        patch_file_path = extract_patch_file_path(patch_file_path)

        print("retrieved", len(retrieved_files))
        if not patch_file_path:
            print(f"[Warning] Could not extract patch file in item {i}")
            continue

        metrics = evaluate_retrieval(patch_file_path, retrieved_files)
        metrics["patch_file"] = patch_file_path
        metrics["instance_id"] = pred.get("instance_id")
        all_metrics.append(metrics)

    # Aggregate stats
    mrr = mean(m["reciprocal_rank"] for m in all_metrics)
    topk_stats = {
        k: mean([m[k] for m in all_metrics])
        for k in ["top_1", "top_3", "top_5", "top_10"]
    }

    print(f"\n=== Overall Stats ===")
    print(f"Total Samples Evaluated: {len(all_metrics)}")
    print(f"MRR: {mrr:.3f}")
    for k, v in topk_stats.items():
        print(f"{k.upper()}: {v:.3f}")

    return all_metrics


# === Example Usage ===
if __name__ == "__main__":
    # Load predictions (replace with actual file or list)
    predictions = read_jsonl_file(
        "results/rq1-1/locagent/locagent_claude-4-6_loc_outputs_normalized.jsonl"
    )

    print("\n\n")
    print("Evaluating code chunking with path retrieval predictions...")
    results = evaluate_predictions(predictions)

    # Optionally save results
    # with open("ranking_eval_results_k50.json", "w") as f:
    #     json.dump(results, f, indent=2)
