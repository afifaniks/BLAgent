# This script compares the localization results of two methods (e.g., BlAgent vs Agentless)
# to analyze cases where one method resolves an instance but the other does not.
# It checks the percentage of such instances where correct localization led to successful repair."""import json

import re

from datasets import load_dataset


def run_analysis(prediction_1, prediction_2, localization_data):
    def extract_patch_file_path(patch_str):
        """Extract the file path that was patched from the diff."""
        match = re.search(r"^diff --git a/(.*?) b/\1", patch_str, re.MULTILINE)
        if match:
            return match.group(1)
        fallback = re.search(r"^\+\+\+ b/(.+)", patch_str, re.MULTILINE)
        if fallback:
            return fallback.group(1)
        return None

    with open(prediction_1, "r") as f:
        prediction_1 = json.load(f)

    with open(prediction_2, "r") as f:
        prediction_2 = json.load(f)

    # read jsonl file
    with open(localization_data, "r") as f:
        lines = f.readlines()
        agentless_file_locs_data = [json.loads(line) for line in lines]

    uncommon = set(prediction_1["resolved_ids"]) - set(prediction_2["resolved_ids"])

    filtered_file_locs = [
        item for item in agentless_file_locs_data if item["instance_id"] in uncommon
    ]

    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Get all instance_ids from the dataset
    instance_ids = [example["instance_id"] for example in dataset]

    # Filter to only include those in uncommon
    filtered_dataset = [item for item in dataset if item["instance_id"] in uncommon]

    patch_dict = {}
    # Get patch files locations
    for item in filtered_dataset:
        instance_id = item["instance_id"]
        patch_file = extract_patch_file_path(item.get("patch"))
        patch_dict[instance_id] = patch_file

    print("Instances resolved by first method but not by second:", len(uncommon))

    count = 0
    for item in filtered_file_locs:
        instance_id = item["instance_id"]
        file_locs = item["found_files"][:3]
        patch_file = patch_dict.get(instance_id)

        if patch_file not in file_locs:
            count += 1
            # print(f"\nInstance ID: {instance_id}")
            # print(f"Expected Patch File: {patch_file}")
            # print(f"Agentless Found Files: {file_locs}")

    print(
        f"\nTotal instances where ground truth file is not in Top-3 candidates: {count}/{len(uncommon)}, Percentage: {count/len(uncommon)*100:.2f}%"
    )

    return count, len(uncommon), count / len(uncommon) * 100


agentless_localization_data = "results/rq2-1/repair_results/agentless_locs.jsonl"
blagent_localization_data = "results/rq2-1/repair_results/blagent_locs.jsonl"

# Run 1
blagent_run_1 = "results/rq2-1/repair_results/run_1/blagent/blagent_majority_voting_and_regression_and_reproduction_evaluation.json"
agentless_run_1 = "results/rq2-1/repair_results/run_1/agentless/agentless_majority_voting_and_regression_and_reproduction_evaluation.json"

# Run 2
blagent_run_2 = "results/rq2-1/repair_results/run_2/blagent/blagent_majority_voting_and_regression_and_reproduction_evaluation.json"
agentless_run_2 = "results/rq2-1/repair_results/run_2/agentless/agentless_majority_voting_and_regression_and_reproduction_evaluation.json"


# Run 3
blagent_run_3 = "results/rq2-1/repair_results/run_3/blagent/blagent_majority_voting_and_regression_and_reproduction_evaluation.json"
agentless_run_3 = "results/rq2-1/repair_results/run_3/agentless/agentless_majority_voting_and_regression_and_reproduction_evaluation.json"


run_analysis(blagent_run_1, agentless_run_1, agentless_localization_data)
run_analysis(blagent_run_2, agentless_run_2, agentless_localization_data)
run_analysis(blagent_run_3, agentless_run_3, agentless_localization_data)

print("================")
run_analysis(agentless_run_1, blagent_run_1, blagent_localization_data)
run_analysis(agentless_run_2, blagent_run_2, blagent_localization_data)
run_analysis(agentless_run_3, blagent_run_3, blagent_localization_data)
