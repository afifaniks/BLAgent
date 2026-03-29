import argparse
import json
import random
import time
from pathlib import Path
from pprint import pprint

import tiktoken
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from blagent.agent import prompts
from blagent.chroma.chroma_store import get_splitter, load_or_build_vector_store
from blagent.util.code_pruner import CodePruner
from blagent.util.code_util import get_code_text_from_path

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agentic reranking with configurable model and context settings."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:120b",
        help="LLM model name.",
    )
    parser.add_argument(
        "--max_files_to_rerank",
        type=int,
        default=10,
        help="Maximum number of files to rerank.",
    )
    parser.add_argument(
        "--num_chunks_per_file",
        type=int,
        default=5,
        help="Number of chunks to keep per file.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Path to save results JSON. If omitted, a default path is generated.",
    )
    parser.add_argument(
        "--function_level",
        action="store_true",
        help="Whether to perform function-level localization (only rank files, not methods).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source localization results JSON.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./swebench_data"),
        help="Path to SWE-bench data directory.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("./repo_data"),
        help="Path to local repository root.",
    )
    parser.add_argument(
        "--chroma_root",
        type=Path,
        default=Path("chroma_data_prune"),
        help="Path to Chroma DB root directory.",
    )
    parser.add_argument(
        "--embed_model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=124000,
        help="Maximum context window in tokens.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum number of retries for failed generations.",
    )

    return parser.parse_args()


args = parse_args()


model = args.model

if "claude" in model.lower():
    llm = ChatAnthropic(model=model)
else:
    llm = ChatOllama(model=model)

MAX_CONTEXT_TOKENS = args.max_context_tokens
MAX_RETRIES = args.max_retries
MAX_FILES_TO_RERANK = args.max_files_to_rerank
NUM_CHUNKS_PER_FILE = args.num_chunks_per_file
FUNCTION_LEVEL = args.function_level
SKIPPED_ENTRIES = []

RESULTS_FILE = args.results_path or (
    f"./retrieval_results/"
    f"agentic_rerank_{model}_pruned_context_"
    f"{NUM_CHUNKS_PER_FILE}chunk{MAX_FILES_TO_RERANK}files.json"
)

source = args.source

DATA_DIR = args.data_dir
REPO_ROOT = args.repo_root
CHROMA_ROOT = args.chroma_root
embed_model_name = args.embed_model_name
embed_model_kwargs = {"device": "cuda", "trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": False}

splitter = get_splitter("code")
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs=embed_model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True,
)

if FUNCTION_LEVEL:
    print(
        "Function level localization enabled: LLM will also rank methods within files, not just files."
    )
else:
    print("File level localization enabled: LLM will only rank files, not methods.")


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
        f"[FILE] {file_path}: Code of size {count_tokens(code_text)} has been reduced to {count_tokens(pruned_code)}"
    )

    return pruned_code


def build_aggregated_code(
    vector_store, entry, file_paths, repo_content, k, use_pruner=True
):
    aggregated = ""
    for file_path in file_paths:
        try:
            if use_pruner:
                code_text = prune_code_context(
                    vector_store,
                    entry,
                    repo_content=repo_content,
                    file_path=file_path,
                    k=k,
                )

                # with open("debug_pruned_code.txt", "w") as f:
                #     f.write(code_text)
                #     exit()
            else:
                code_text = get_code_text_from_path(repo_content, file_path.split("/"))
        except Exception as e:
            print(f"❌ Error retrieving {file_path}: {e}")
            code_text = "<Error retrieving file content>"
        aggregated += f"### File: {file_path} ###\n" f"{code_text}\n\n"
    return aggregated


def rerank_files_llm(vector_store, file_paths, entry, repo_content, llm):
    prompt = build_prompt(vector_store, file_paths, entry, repo_content)

    num_tokens = count_tokens(prompt)
    print(f"Number of tokens in {len(file_paths)} files: {num_tokens}")

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant that reranks code files.",
            },
            {"role": "user", "content": prompt},
        ]
    )

    try:
        content = response.content

        output_tokens = count_tokens(content)

        # Extract JSON block from markdown code fence if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        # token_usage = {
        #     "prompt_tokens": num_tokens,
        #     "output_tokens": output_tokens,
        #     "total_tokens": num_tokens + output_tokens,
        # }
        token_usage = {
            "prompt_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"],
        }

        parsed = json.loads(content)
        return parsed["ranked_files"], token_usage
    except Exception:
        print("❌ Failed to parse LLM output")
        print(response.content)
        raise


def build_prompt(vector_store, file_paths, entry, repo_content):
    aggregated_code_text = build_aggregated_code(
        vector_store,
        entry,
        file_paths,
        repo_content,
        k=NUM_CHUNKS_PER_FILE,
    )

    if FUNCTION_LEVEL:
        prompt = prompts.FUNCTION_LEVEL_LOCALIZATION_PROMPT.format(
            problem_statement=entry["problem_statement"],
            aggregated_code_text=aggregated_code_text,
            num_file_paths=len(file_paths),
        )
    else:
        prompt = prompts.EVIDENCE_ANCHORED_RERANKER_PROMPT.format(
            problem_statement=entry["problem_statement"],
            aggregated_code_text=aggregated_code_text,
            num_file_paths=len(file_paths),
        )

    return prompt


def load_repo_content(swe_index):
    """Load repository content for a given SWE index."""
    with open(f"./swebench_data/swebench_{swe_index}_py.json", "r") as f:
        return json.load(f)


def get_sorted_files(entry):
    """Sort files by their ranked scores."""
    scores = entry["ranked_scores"]
    return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def perform_single_pass_rerank(vector_store, ranked_files, entry, repo_content, llm):
    """Perform single-pass reranking."""
    print("\n✅ Using single-pass reranking")
    return rerank_files_llm(vector_store, ranked_files, entry, repo_content, llm)


def perform_tournament_rerank(
    vector_store, ranked_files, entry, repo_content, llm, use_pruner=True
):
    """Perform tournament-style reranking for large contexts."""
    print("\n⚠️ Context too large — using tournament reranking")

    # random.shuffle(ranked_files)

    # mid = len(ranked_files) // 2
    # first_half = ranked_files[:mid]
    # second_half = ranked_files[mid:]

    already_used_indices = []
    first_split = []
    total_token_count = 0

    print("\n=== Preparing Tournament Split 1 ===")
    for i, file_path in enumerate(ranked_files):
        print(f"Evaluating file: {file_path}")
        try:
            if use_pruner:
                print("Pruning context")
                code_text = prune_code_context(
                    vector_store,
                    entry,
                    repo_content=repo_content,
                    file_path=file_path,
                    k=NUM_CHUNKS_PER_FILE,
                )
            else:
                code_text = get_code_text_from_path(repo_content, file_path.split("/"))
        except Exception as e:
            print(f"❌ Error retrieving {file_path}: {e}")
            code_text = "<Error retrieving file content>"
        file_token_count = count_tokens(code_text)
        if total_token_count + file_token_count > MAX_CONTEXT_TOKENS:
            continue
        first_split.append(file_path)
        total_token_count += file_token_count
        already_used_indices.append(i)

    print("\n=== Preparing Tournament Split 2 ===")
    second_split = []
    total_token_count = 0
    for i, file_path in enumerate(ranked_files):
        if i in already_used_indices:
            continue
        try:
            if use_pruner:
                print("Pruning context")
                code_text = prune_code_context(
                    vector_store,
                    entry,
                    repo_content=repo_content,
                    file_path=file_path,
                    k=NUM_CHUNKS_PER_FILE,
                )
            else:
                code_text = get_code_text_from_path(repo_content, file_path.split("/"))
        except Exception as e:
            print(f"❌ Error retrieving {file_path}: {e}")
            code_text = "<Error retrieving file content>"
        file_token_count = count_tokens(code_text)
        if total_token_count + file_token_count > MAX_CONTEXT_TOKENS:
            continue
        second_split.append(file_path)
        total_token_count += file_token_count

    print("\n=== Round 1: First Split ===")
    print(first_split)
    ranked_first, token_usage = rerank_files_llm(
        vector_store, first_split, entry, repo_content, llm
    )

    print("\n=== Round 1: Second Split ===")
    print(second_split)
    ranked_second, token_usage = rerank_files_llm(
        vector_store, second_split, entry, repo_content, llm
    )

    tournament_survivors = []
    total_token_count = 0
    for file_path in ranked_first[:3] + ranked_second[:3]:
        try:
            if use_pruner:
                print("Pruning context")
                code_text = prune_code_context(
                    vector_store,
                    entry,
                    repo_content=repo_content,
                    file_path=file_path,
                    k=NUM_CHUNKS_PER_FILE,
                )
            else:
                code_text = get_code_text_from_path(repo_content, file_path.split("/"))
        except Exception as e:
            print(f"❌ Error retrieving {file_path}: {e}")
            code_text = "<Error retrieving file content>"
        file_token_count = count_tokens(code_text)
        if total_token_count + file_token_count > MAX_CONTEXT_TOKENS:
            continue
        tournament_survivors.append(file_path)
        total_token_count += file_token_count

    print("\n=== Final Candidates ===")
    pprint(tournament_survivors)

    return rerank_files_llm(
        vector_store, tournament_survivors, entry, repo_content, llm
    )


def ensure_complete_ranking(final_ranking, ranked_files):
    """
    Ensure all original files are in the final ranking, preserving order.

    Assumptions:
      - final_ranking: list[dict] where each dict has exactly one key: {filepath: ...}
      - ranked_files: list[str] of original filepaths
    """
    seen = set()
    result = []

    if isinstance(final_ranking, list) and all(
        isinstance(item, str) for item in final_ranking
    ):
        print(
            "File level localization detected. Converting to expected format with empty method lists."
        )
        final_ranking = [{f: []} for f in final_ranking]

    # Keep LLM-ranked order first
    for item in final_ranking:
        if not isinstance(item, dict):
            raise TypeError(
                f"final_ranking items must be dicts, got: {type(item)} -> {item}"
            )
        if len(item) != 1:
            raise ValueError(
                f"Each final_ranking dict must have exactly 1 key, got {len(item)} -> {item}"
            )

        key = next(iter(item.keys()))
        if key not in seen:
            seen.add(key)
            result.append(item)

    # Append missing files in original order
    for f in ranked_files:
        if f not in seen:
            seen.add(f)
            result.append({f: []})

    return result


def process_entry_with_retry(entry, llm, max_retries=MAX_RETRIES):
    """Process a single entry with retry logic."""
    for attempt in range(1, max_retries + 1):
        # try:
        print(
            f"\n=== Processing index {entry['swe_data_index']} (Attempt {attempt}/{max_retries}) ==="
        )
        print(f"Loading vectore store...")
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

        sorted_keys = get_sorted_files(entry)
        entry["ranked_files"] = sorted_keys

        repo_content = load_repo_content(entry["swe_data_index"])
        ranked_files = entry["ranked_files"]

        # Check total context size
        full_aggregated_code = build_aggregated_code(
            vector_store,
            entry,
            ranked_files[:MAX_FILES_TO_RERANK],
            repo_content,
            k=NUM_CHUNKS_PER_FILE,
        )
        full_prompt_tokens = count_tokens(
            entry["problem_statement"] + full_aggregated_code
        )
        print(f"\nTotal tokens if single-pass: {full_prompt_tokens}")

        # Choose reranking strategy
        if full_prompt_tokens <= MAX_CONTEXT_TOKENS:
            final_ranking, token_usage = perform_single_pass_rerank(
                vector_store,
                ranked_files[:MAX_FILES_TO_RERANK],
                entry,
                repo_content,
                llm,
            )
        else:
            final_ranking, token_usage = perform_tournament_rerank(
                vector_store,
                ranked_files[:MAX_FILES_TO_RERANK],
                entry,
                repo_content,
                llm,
            )

        # Ensure complete ranking
        final_ranking = ensure_complete_ranking(final_ranking, ranked_files)

        print("\n=== Final Ranking ===")
        pprint(final_ranking)
        print("\nActual Patch File:", entry["patch_file"])
        print("\nToken Usage:")
        pprint(token_usage)

        entry["final_reranked_files"] = final_ranking
        entry["token_usage"] = token_usage
        return entry


def save_intermediate_results(results, output_file):
    """Save intermediate results to file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    with open(source, "r") as f:
        all_retrieved_docs = json.load(f)

    try:
        with open(RESULTS_FILE, "r") as f:
            all_reranked_results = json.load(f)
    except FileNotFoundError:
        all_reranked_results = []

    # filtered_id = [115, 111, 107, 102]
    # all_retrieved_docs = [
    #     entry for entry in all_retrieved_docs if entry["swe_data_index"] in filtered_id
    # ]

    # Convert to set of processed indexes for quick lookup
    processed_indexes = {entry["swe_data_index"] for entry in all_reranked_results}

    for entry in all_retrieved_docs:
        if entry["swe_data_index"] in processed_indexes:
            print(f"Skipping index {entry['swe_data_index']} (already processed).")
            continue

        processed_entry = process_entry_with_retry(entry, llm)

        if processed_entry:
            all_reranked_results.append(processed_entry)
            save_intermediate_results(all_reranked_results, RESULTS_FILE)
        else:
            print(f"⚠️ Skipping entry {entry.get('swe_data_index', 'unknown')}")
