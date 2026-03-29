import json
import os
import re
from pprint import pprint

from datasets import load_dataset


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


# Load the main SWE-bench dataset
swebench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

patch_map = {}
for swe_data_index in range(len(swebench)):
    swe_data = swebench[swe_data_index]
    patch = swe_data["patch"]
    instance_id = swe_data["instance_id"]
    patch_file_path = extract_patch_file_path(patch)

    patch_map[instance_id] = patch_file_path


agentless_file = "/local/home/amamun/projects/Agentless/agentless_results/swe-bench-lite/file_level_combined/combined_locs.jsonl"

loc_outputs = read_jsonl_file(agentless_file)

# Add the patch file path to each loc output
for loc_output in loc_outputs:
    instance_id = loc_output["instance_id"]
    if instance_id in patch_map:
        loc_output["patch_file_path"] = patch_map[instance_id]
    else:
        loc_output["patch_file_path"] = None

# Save the updated loc outputs to a new JSONL file
output_file = "agentless_gpt_oss_outputs_with_patch.jsonl"
with open(output_file, "w") as file:
    for loc_output in loc_outputs:
        file.write(json.dumps(loc_output) + "\n")
