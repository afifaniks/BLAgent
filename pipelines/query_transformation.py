import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pprint import pprint

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from blagent.client.ollama_client import OllamaClient
from blagent.client.openai_client import OpenAIClient
from blagent.retriever.code_retriever import CodeRetriever
from blagent.util.code_util import extract_patch_file_path, get_code_text_from_path
from blagent.util.query_augmentation import QueryAugmentor

load_dotenv()
# Arguments
parser = argparse.ArgumentParser(
    description="SWE-bench Retrieval with Query Augmentation"
)

parser.add_argument(
    "--data_dir",
    type=str,
    default="./swebench_data",
    help="Directory containing SWE-bench data.",
)

parser.add_argument(
    "--chroma_root",
    type=str,
    default="./chroma_data_code_with_path",
    help="Root directory for Chroma vector stores.",
)

parser.add_argument(
    "--repo_root",
    type=str,
    default="./repo_data",
    help="Root directory for repository data.",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature setting for query augmentation.",
)

parser.add_argument(
    "--llm",
    type=str,
    default="gpt-oss:120b",
    help="Model name for the LLM.",
)
parser.add_argument(
    "--results_path",
    type=str,
    default="retrieval_results/augmented_query_results.json",
    help="Path to store retrieval results.",
)

args = parser.parse_args()

DATA_DIR = args.data_dir
REPO_ROOT = args.repo_root
CHROMA_ROOT = args.chroma_root
TEMPERATURE = args.temperature
llm = args.llm
results_path = args.results_path

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

llm_client = OllamaClient(base_url=os.environ["OLLAMA_HOST"])

query_augmentor = QueryAugmentor(llm_client, model_name=llm)

code_retriever = CodeRetriever(DATA_DIR, REPO_ROOT, CHROMA_ROOT)

swebench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")


def augment_and_retrieve(
    query_augmentor,
    code_retriever,
    swe_data_index,
    swebench,
    problem_statement,
    n=2,
    max_retries=5,
    retry_delay=2,  # in seconds
    max_files=15,
):
    all_retrieved_files = []
    augmentated_queries = []

    def safe_retrieve(aug_query):
        for attempt in range(max_retries):
            try:
                result = code_retriever.retrieve_similar_code(
                    swe_data_index, swebench, augmented_query=aug_query
                )
                return result["retrieved_files"]
            except OSError as e:
                print(f"OSError on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(retry_delay)
        print(f"Failed after {max_retries} retries. Skipping query.")
        return []

    # Base retrieval without augmentation
    # print("Running base retrieval...")
    # base_retrievals = safe_retrieve(problem_statement)
    # print("Base retrievals:", base_retrievals)
    # all_retrieved_files.append(base_retrievals[:max_files])

    # Augmented queries
    for i in tqdm(range(n), desc="Augmenting..."):
        try:
            augmented_query = query_augmentor.augment(
                problem_statement, temperature=0.7, type=i
            )
            print()
            print("Type: " + str(i) + " Augmented query: " + augmented_query)
            print("-" * 20)
            retrieved = safe_retrieve(augmented_query)
            all_retrieved_files.append(retrieved[:max_files])
            augmentated_queries.append(augmented_query)
        except Exception as e:
            print(f"Query augmentation failed on iteration {i}: {e}")

    # # Majority voting
    # file_counts = Counter([item for sublist in all_retrieved_files for item in sublist])
    # print("File counts:", file_counts)
    # majority_ranked_files = [file for file, _ in file_counts.most_common(max_files)]

    return augmentated_queries, all_retrieved_files


results = []

if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)

for index in tqdm(range(len(dataset))):
    swe_data = dataset[index]
    swe_data_index = index

    if swe_data_index in [result["swe_data_index"] for result in results]:
        print(f"Skipping already processed index: {swe_data_index}")
        continue
    problem_statement = swe_data["problem_statement"]
    patch_file = extract_patch_file_path(swe_data["patch"])

    with open(f"./swebench_data/swebench_{swe_data_index}_py.json", "r") as f:
        failed_repo_content = json.load(f)

    file_paths = patch_file.split("/")

    # augmented_query = query_augmentor.augment(failed_data["problem_statement"])
    # retrieved_files = code_retriever.retrieve_similar_code(
    #     swe_data_index, swebench, augmented_query=augmented_query
    # )["retrieved_files"]
    augmented_queries, retrieved_files = augment_and_retrieve(
        query_augmentor,
        code_retriever,
        swe_data_index,
        swebench,
        swe_data["problem_statement"],
    )
    is_patch_in_top10_type0 = patch_file in retrieved_files[0][:10]
    is_patch_in_top10_type1 = patch_file in retrieved_files[1][:10]

    # Print for inspection
    print("Problem statement\n" + problem_statement)
    print("-" * 20)
    print("Retrieved files:")
    pprint(retrieved_files)
    print("-" * 20)
    print("Patch file:", patch_file)
    print("Is Patch in Top-10-T0:", is_patch_in_top10_type0)
    print("Is Patch in Top-10-T1", is_patch_in_top10_type1)

    print("=" * 20)

    # Append result to list
    results.append(
        {
            "swe_data_index": swe_data_index,
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "augmented_query": augmented_queries,
            "patch_file": patch_file,
            "is_patch_in_top10_t0": is_patch_in_top10_type0,
            "is_patch_in_top10_t1": is_patch_in_top10_type1,
            "retrieved_files_t0": retrieved_files[0],
            "retrieved_files_t1": retrieved_files[1],
        }
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(results_path) as f:
        data = json.load(f)

    ctr = 0
    for d in data:
        if d["is_patch_in_top10_t0"] or d["is_patch_in_top10_t1"]:
            ctr += 1

    print(ctr, "/", len(data))
