import argparse
import json

from datasets import load_dataset

# Load SWE-bench Lite (test split)
swebench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

parser = argparse.ArgumentParser()
parser.add_argument("--blagent_output", type=str)
parser.add_argument("--converted_output", type=str)
args = parser.parse_args()

blagent_output = args.blagent_output
converted_output = args.converted_output

with open(blagent_output, "r") as f:
    ragent_predictions = json.load(f)

# Write JSONL
with open(converted_output, "w") as out_f:
    for idx, data in enumerate(swebench):
        instance_id = data["instance_id"]
        found_files = ragent_predictions[idx]["final_reranked_files"]
        patch_file = ragent_predictions[idx]["patch_file"]
        swe_data_index = ragent_predictions[idx]["swe_data_index"]
        additional_artifact_loc_file = {}
        file_traj = {}

        # Build dict for JSONL line
        record = {
            "swe_data_index": swe_data_index,
            "instance_id": instance_id,
            "found_files": found_files,
            "patch_file": patch_file,
            "additional_artifact_loc_file": additional_artifact_loc_file,
            "file_traj": file_traj,
        }

        out_f.write(json.dumps(record) + "\n")
