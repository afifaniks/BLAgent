import json
import os

from datasets import load_dataset
from tqdm import tqdm

from blagent.git.repository import get_project_structure_from_scratch
from blagent.util.preprocess_data import filter_none_python, filter_out_test_files

# Load the main SWE-bench dataset
swebench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")


for i in tqdm(
    range(len(swebench)),
    desc="Processing SWE-bench dataset",
    total=len(swebench),
    colour="green",
):
    swe_data = swebench[i]

    patch = swe_data["patch"]
    problem_statement = swe_data["problem_statement"]
    DATA_DIR = "./swebench_data"
    os.makedirs(DATA_DIR, exist_ok=True)
    RAW_JSON_PATH = f"swebench_{i}_raw.json"
    PY_JSON_PATH = f"swebench_{i}_py.json"

    RAW_JSON_PATH = os.path.join(DATA_DIR, RAW_JSON_PATH)
    PY_JSON_PATH = os.path.join(DATA_DIR, PY_JSON_PATH)

    if not os.path.exists(PY_JSON_PATH):
        # Get data
        project_structure = get_project_structure_from_scratch(
            swe_data["repo"],
            swe_data["base_commit"],
            swe_data["instance_id"],
            "./repo_data",
            cleanup=True,
        )

        with open(RAW_JSON_PATH, "w") as f:
            json.dump(project_structure, f, indent=2)

        structure = project_structure["structure"]
        filter_none_python(structure)
        filter_out_test_files(structure)

        with open(PY_JSON_PATH, "w") as f:
            json.dump(structure, f, indent=2)
    else:
        print(f"Skipping {PY_JSON_PATH} as it already exists.")
