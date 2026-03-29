import json
import re
from statistics import mean


def extract_patch_file_path(patch_str):
    """Extract the file path that was patched from the diff."""
    match = re.search(r"^diff --git a/(.*?) b/\1", patch_str, re.MULTILINE)
    if match:
        return match.group(1)
    fallback = re.search(r"^\+\+\+ b/(.+)", patch_str, re.MULTILINE)
    if fallback:
        return fallback.group(1)
    return None


def evaluate_retrieval(patch_file_path, retrieved_files, top_k=(1, 3, 5, 10, 15)):
    result = {}
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


def find_incorrect_files(predictions, ground_truth):
    incorrect_files = []
    for i, pred in enumerate(predictions):
        patch = pred.get("patch", "")
        retrieved_files = pred.get("retrieved_files", [])[:10]
        patch_file_path = extract_patch_file_path(patch)

        if not patch_file_path:
            print(f"[Warning] Could not extract patch file in item {i}")
            continue

        if patch_file_path not in ground_truth:
            incorrect_files.append(
                {
                    "index": i,
                    "patch_file": patch_file_path,
                    "retrieved_files": retrieved_files,
                }
            )

    return incorrect_files


def evaluate_predictions(predictions):
    all_metrics = []
    for i, pred in enumerate(predictions):
        patch = pred.get("patch", "")
        retrieved_files = pred.get("retrieved_files", [])[:10]
        patch_file_path = extract_patch_file_path(patch)

        if not patch_file_path:
            print(f"[Warning] Could not extract patch file in item {i}")
            continue

        if patch_file_path not in retrieved_files:
            metrics = {}
            metrics["patch_file"] = patch_file_path
            metrics["problem_statement"] = pred.get("problem_statement", "")
            metrics["swe_data_index"] = pred.get("swe_data_index", i)
            metrics["retrieved_files"] = retrieved_files
            all_metrics.append(metrics)

    return all_metrics


# === Example Usage ===
if __name__ == "__main__":
    # Load predictions (replace with actual file or list)
    with open(
        "/work/disa_lab/afif/projects/rag_fix/retrieval_results_code_test_with_path_k50.json", "r"
    ) as f:
        predictions = json.load(f)

    print("Evaluating code splitter retrieval predictions...")
    results = evaluate_predictions(predictions)

    with open("failed_cases_k50.json", "w") as f:
        json.dump(results, f, indent=2)
