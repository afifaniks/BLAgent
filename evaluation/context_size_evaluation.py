import csv
import json
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from blagent.chroma.chroma_store import get_splitter, load_or_build_vector_store
from blagent.util.code_pruner import CodePruner
from blagent.util.code_util import get_code_text_from_path

load_dotenv()

MAX_RETRIES = 10
SKIPPED_ENTRIES = []
SOURCE = "../"

DATA_DIR = Path("./swebench_data")
REPO_ROOT = Path("./repo_data")
CHROMA_ROOT = Path("chroma_data_prune")

OUTPUT_CSV = "token_usage_file_chunk_sweep.csv"

FILE_VARIATIONS = [5, 10]
CHUNK_VARIATIONS = [5, 10, 20]

embed_model_name = "nomic-ai/nomic-embed-text-v1"
embed_model_kwargs = {"device": "cuda", "trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": False}

splitter = get_splitter("code")
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs=embed_model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True,
)


# -----------------------------
# Token counting
# -----------------------------
def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def prune_code_context(vector_store, entry, repo_content, file_path, k):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k, "filter": {"relative_path": file_path}}
    )
    retrieved_docs = retriever.get_relevant_documents(entry["augmented_query"][0])
    chunks = [doc.page_content for doc in retrieved_docs]

    code_text = get_code_text_from_path(repo_content, file_path.split("/"))
    code_pruner = CodePruner(code_text, chunks)
    pruned_code = code_pruner.prune()

    print(
        f"[FILE] {file_path}: Code of size {count_tokens(code_text)} "
        f"has been reduced to {count_tokens(pruned_code)} using top-{k} chunks"
    )

    return pruned_code


def build_aggregated_code(
    vector_store,
    entry,
    file_paths,
    repo_content,
    max_chunks_per_file,
    use_pruner=True,
):
    aggregated = ""

    for file_path in file_paths:
        try:
            if use_pruner:
                code_text = prune_code_context(
                    vector_store=vector_store,
                    entry=entry,
                    repo_content=repo_content,
                    file_path=file_path,
                    k=max_chunks_per_file,
                )
            else:
                code_text = get_code_text_from_path(repo_content, file_path.split("/"))
        except Exception as e:
            print(f"❌ Error retrieving {file_path}: {e}")
            code_text = "<Error retrieving file content>"

        aggregated += f"### File: {file_path} ###\n{code_text}\n\n"

    return aggregated


def load_repo_content(swe_index):
    with open(f"./swebench_data/swebench_{swe_index}_py.json", "r") as f:
        return json.load(f)


def get_sorted_files(entry):
    scores = entry["ranked_scores"]
    return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def compute_prompt_tokens_for_setting(
    entry,
    vector_store,
    repo_content,
    max_files,
    max_chunks_per_file,
):
    ranked_files = get_sorted_files(entry)
    selected_files = ranked_files[:max_files]

    aggregated_code = build_aggregated_code(
        vector_store=vector_store,
        entry=entry,
        file_paths=selected_files,
        repo_content=repo_content,
        max_chunks_per_file=max_chunks_per_file,
    )

    total_prompt_text = entry["problem_statement"] + aggregated_code
    total_prompt_tokens = count_tokens(total_prompt_text)

    print(
        f"[TOKENS] index={entry['swe_data_index']} | "
        f"MAX_FILES={max_files} | MAX_CHUNKS_PER_FILE={max_chunks_per_file} | "
        f"tokens={total_prompt_tokens}"
    )

    return total_prompt_tokens


def process_entry(entry):
    print(f"\n=== Processing index {entry['swe_data_index']} ===")
    print("Loading vector store...")

    vector_store = load_or_build_vector_store(
        entry,
        entry["swe_data_index"],
        embed_model,
        splitter,
        data_dir=DATA_DIR,
        chroma_root=CHROMA_ROOT,
        repo_root=REPO_ROOT,
        include_path_in_chunk=True,
    )

    repo_content = load_repo_content(entry["swe_data_index"])

    results = []
    for max_files in FILE_VARIATIONS:
        for max_chunks_per_file in CHUNK_VARIATIONS:
            try:
                token_count = compute_prompt_tokens_for_setting(
                    entry=entry,
                    vector_store=vector_store,
                    repo_content=repo_content,
                    max_files=max_files,
                    max_chunks_per_file=max_chunks_per_file,
                )

                results.append(
                    {
                        "swe_data_index": entry["swe_data_index"],
                        "max_files": max_files,
                        "max_chunks_per_file": max_chunks_per_file,
                        "prompt_tokens": token_count,
                    }
                )
            except Exception as e:
                print(
                    f"❌ Failed for index={entry['swe_data_index']}, "
                    f"MAX_FILES={max_files}, MAX_CHUNKS_PER_FILE={max_chunks_per_file}: {e}"
                )
                results.append(
                    {
                        "swe_data_index": entry["swe_data_index"],
                        "max_files": max_files,
                        "max_chunks_per_file": max_chunks_per_file,
                        "prompt_tokens": None,
                    }
                )

    return results


def write_results_csv(rows, output_csv):
    fieldnames = [
        "swe_data_index",
        "max_files",
        "max_chunks_per_file",
        "prompt_tokens",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\n=== Summary by setting ===")
    for max_files in FILE_VARIATIONS:
        for max_chunks_per_file in CHUNK_VARIATIONS:
            subset = [
                r["prompt_tokens"]
                for r in rows
                if r["max_files"] == max_files
                and r["max_chunks_per_file"] == max_chunks_per_file
                and r["prompt_tokens"] is not None
            ]

            if not subset:
                print(
                    f"MAX_FILES={max_files}, "
                    f"MAX_CHUNKS_PER_FILE={max_chunks_per_file} -> no valid entries"
                )
                continue

            avg_tokens = sum(subset) / len(subset)
            min_tokens = min(subset)
            max_tokens = max(subset)

            print(
                f"MAX_FILES={max_files}, MAX_CHUNKS_PER_FILE={max_chunks_per_file} | "
                f"avg={avg_tokens:.2f}, min={min_tokens}, max={max_tokens}, n={len(subset)}"
            )


if __name__ == "__main__":
    with open(SOURCE, "r") as f:
        all_retrieved_docs = json.load(f)

    all_rows = []

    for entry in all_retrieved_docs:
        try:
            entry_rows = process_entry(entry)
            all_rows.extend(entry_rows)
        except Exception as e:
            print(f"❌ Failed to process entry {entry.get('swe_data_index')}: {e}")

    write_results_csv(all_rows, OUTPUT_CSV)
    print_summary(all_rows)

    print(f"\nCSV written to: {OUTPUT_CSV}")
