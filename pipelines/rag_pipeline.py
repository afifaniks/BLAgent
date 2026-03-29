# rag_pipeline.py
import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from datasets import load_dataset
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from blagent.chroma.chroma_store import get_splitter, load_or_build_vector_store

DATA_DIR = Path("./swebench_data")
REPO_ROOT = Path("./repo_data")
CHROMA_ROOT = Path("chroma_data")

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
LLM_MODEL = "gpt-oss:120b"
EMBED_MODEL_KWARGS = {"device": "cuda", "trust_remote_code": True}
ENCODE_KWARGS = {"normalize_embeddings": False}
INCLUDE_PATH_IN_CHUNK = False


def make_embed_model(
    model_name: str = EMBED_MODEL,
    model_kwargs: dict = EMBED_MODEL_KWARGS,
    encode_kwargs: dict = ENCODE_KWARGS,
):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
    )


def get_retriever_from_store(store: Chroma, k: int = 50):
    return store.as_retriever(search_kwargs={"k": k})


def rag_rank_files(
    llm: Callable[[str], str], question: str, context_docs: List[Document]
) -> List[str]:
    """
    Use an LLM to analyze retrieved code snippets and rank files
    most likely related to the described bug.
    """
    context = "\n\n---\n\n".join(
        [
            f"[FILE] {d.metadata.get('relative_path', '')}\n{d.page_content[:2000]}"
            for d in context_docs[:15]
        ]
    )
    prompt = f"""
You are an expert software debugging assistant.

Your goal: Identify which retrieved files are most likely to contain the bug described below.

Bug Report:
{question}

Code Context:
{context}

Instructions:
1. Carefully analyze the bug report and snippets.
2. Respond ONLY with a JSON array of file paths, ordered from most to least relevant.
3. If there is no path provided, return an empty JSON array.
4. DO NOT include any explanations or additional text.
5. Return 10 ranked file paths from the context.
6. DO NOT include paths that were not provided.

Example:
["src/core/utils.py", "src/main/model.py", "src/data/loader.py"]
"""

    raw_response = llm(prompt)
    try:
        ranked_files = json.loads(raw_response)
        if isinstance(ranked_files, list) and all(
            isinstance(x, str) for x in ranked_files
        ):
            return ranked_files
        return []
    except Exception:
        return []


def process_single_example(
    swebench_entry: dict,
    index: int,
    splitter,
    embed_model=None,
    use_rag: bool = False,
    llm: Optional[Callable[[str], str]] = None,
    k: int = 50,
) -> Dict:
    if embed_model is None:
        embed_model = make_embed_model()

    vector_store = load_or_build_vector_store(
        swebench_entry,
        index,
        embed_model,
        splitter,
        data_dir=DATA_DIR,
        chroma_root=CHROMA_ROOT,
        repo_root=REPO_ROOT,
        include_path_in_chunk=INCLUDE_PATH_IN_CHUNK,
    )
    # Excluding time taken to encode
    start_time = time.time()

    retriever = get_retriever_from_store(vector_store, k=k)

    problem_statement = swebench_entry.get("problem_statement", "")
    patch = swebench_entry.get("patch", "")

    retrieved_docs = retriever.get_relevant_documents(problem_statement)
    retrieved_paths = []
    for d in retrieved_docs:
        print(d)
        retrieved_paths.append(d.metadata.get("relative_path", ""))

    seen = set()
    retrieved_paths = [p for p in retrieved_paths if not (p in seen or seen.add(p))]

    result = {
        "swe_data_index": index,
        "problem_statement": problem_statement,
        "patch": patch,
        "retrieved_files": retrieved_paths,
    }

    if use_rag and llm is not None:
        try:
            ranked_files = rag_rank_files(llm, problem_statement, retrieved_docs)
            result["rag_ranked_files"] = ranked_files
        except Exception as e:
            result["rag_error"] = str(e)

    time_taken = time.time() - start_time
    result["time_taken_seconds"] = time_taken

    return result


def run_dataset(
    split: str,
    splitter,
    persist_results: Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    start: int = 0,
    end: Optional[int] = None,
    use_rag: bool = False,
    llm: Optional[Callable[[str], str]] = None,
    skip_existing: bool = True,
):
    swebench = load_dataset(dataset_name, split=split)
    n = len(swebench) if end is None else min(end, len(swebench))
    all_results = []
    processed = set()

    if skip_existing:
        if persist_results.exists():
            with persist_results.open("r") as f:
                all_results = json.load(f)
                processed = {r["swe_data_index"] for r in all_results}

    embed_model = make_embed_model()

    for i in tqdm(range(start, n), desc="Processing SWE data"):
        if i in processed:
            continue
        swe_entry = swebench[i]
        res = process_single_example(
            swe_entry, i, splitter, embed_model=embed_model, use_rag=use_rag, llm=llm
        )

        all_results.append(res)
        with persist_results.open("w") as f:
            json.dump(all_results, f, indent=2)


def ollama_ranker(prompt: str) -> str:
    import requests

    response = requests.post(
        os.environ["OLLAMA_HOST"] + "/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
    )
    data = response.json()
    return data.get("response", "").strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splitter",
        type=str,
        default="code",
        choices=["code", "text"],
        help="Type of text splitter to use.",
    )
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Whether to use RAG (Retrieval-Augmented Generation).",
    )
    parser.add_argument(
        "--include_path_in_chunk",
        action="store_true",
        help="Whether to include file path in each text chunk.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        help="LLM function to use for ranking.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./retrieval_results/ranking_results.json",
        help="Path to persist retrieval results.",
    )
    parser.add_argument(
        "--chroma_root",
        type=str,
        help="Root directory for Chroma vector stores.",
    )

    args = parser.parse_args()

    if args.include_path_in_chunk:
        print("[main] including file path in each text chunk")
        INCLUDE_PATH_IN_CHUNK = True

    if args.llm:
        print(f"[main] using LLM model: {args.llm}")
        LLM_MODEL = args.llm

    if args.chroma_root:
        print(f"[main] using Chroma root: {args.chroma_root}")
        CHROMA_ROOT = Path(args.chroma_root)

    persist_results = Path(args.results_path)

    splitter = args.splitter
    run_dataset(
        split="test",
        splitter=get_splitter(splitter),
        persist_results=persist_results,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        start=0,
        end=None,
        use_rag=args.use_rag,
        llm=ollama_ranker,
        skip_existing=True,
    )
