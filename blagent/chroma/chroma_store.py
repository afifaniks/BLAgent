# rag_pipeline.py
import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import tree_sitter_python as tspython
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import CodeSplitter
from tqdm import tqdm
from tree_sitter import Language, Parser

from blagent.git.repository import get_project_structure_from_scratch
from blagent.util.preprocess_data import filter_none_python, filter_out_test_files


def ensure_repo_json(
    swe_entry: dict,
    index: int,
    data_dir: Path,
    repo_root: Path,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    json_name = f"swebench_{index}_py.json"
    json_path = data_dir / json_name
    if not json_path.exists():
        project_structure = get_project_structure_from_scratch(
            swe_entry["repo"],
            swe_entry["base_commit"],
            swe_entry["instance_id"],
            str(repo_root),
            cleanup=True,
        )
        structure = project_structure["structure"]
        filter_none_python(structure)
        filter_out_test_files(structure)
        with json_path.open("w") as f:
            json.dump(structure, f, indent=2)
    return json_path


def split_documents(
    docs: Iterable[Document], splitter, include_path_in_chunk: bool = False
) -> List[Document]:
    output: List[Document] = []

    if isinstance(splitter, RecursiveCharacterTextSplitter):
        return splitter.split_documents(docs)

    for doc in docs:
        try:
            chunks = splitter.split_text(doc.page_content)
            file_path = doc.metadata.get("relative_path", "")
            for idx, chunk in enumerate(chunks):
                if include_path_in_chunk:
                    content = f"[PATH] {file_path}\n[CODE]\n{chunk}"
                else:
                    content = chunk
                meta = {**doc.metadata, "split_index": idx}
                output.append(Document(page_content=content, metadata=meta))
        except Exception as e:
            print(
                f"[split_documents] failed to split {doc.metadata.get('relative_path','')}: {e}"
            )
    return output


def build_chroma_from_documents(
    documents: List[Document], chroma_dir: Path, embed_model
) -> Chroma:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    vect = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=str(chroma_dir),
    )
    vect.persist()
    return vect


def _is_python_path(path: str) -> bool:
    return str(path).endswith(".py")


def collect_code_nodes(
    node: object, path_parts: List[str], repo_name: str
) -> List[Document]:
    documents: List[Document] = []
    relative_path = os.path.join(*path_parts)
    is_python = _is_python_path(relative_path)

    if is_python and isinstance(node, dict):
        try:
            code = "\n".join(node.get("text", []))
            meta = {
                "repo": repo_name,
                "file": relative_path,
                "relative_path": os.path.join(repo_name, relative_path),
                "type": "file",
            }
            documents.append(Document(page_content=code, metadata=meta))
        except Exception as e:
            print(
                f"[collect_code_nodes] error reading {relative_path} in {repo_name}: {e}"
            )
    elif isinstance(node, dict):
        for key, subnode in node.items():
            documents.extend(collect_code_nodes(subnode, path_parts + [key], repo_name))
    return documents


def load_documents_from_json(json_path: Path) -> List[Document]:
    with json_path.open("r") as f:
        data = json.load(f)
    all_docs: List[Document] = []
    for repo_name, repo_content in data.items():
        all_docs.extend(collect_code_nodes(repo_content, [""], repo_name))
    return all_docs


def load_or_build_vector_store(
    swe_entry: dict,
    index: int,
    embed_model,
    splitter,
    data_dir: Path,
    chroma_root: Path,
    repo_root: Path,
    include_path_in_chunk: bool = False,
) -> Chroma:
    repo_json = ensure_repo_json(swe_entry, index, data_dir, repo_root)
    docs = load_documents_from_json(repo_json)
    chroma_dir = chroma_root / f"chroma_repo_index_swe_data{index}"

    if chroma_dir.exists():
        print(f"[load_or_build_vector_store] loading existing Chroma: {chroma_dir}")
        return Chroma(embedding_function=embed_model, persist_directory=str(chroma_dir))

    print("[load_or_build_vector_store] splitting and building vector store...")
    split_docs = split_documents(docs, splitter, include_path_in_chunk)
    print(f"[load_or_build_vector_store] split into {len(split_docs)} chunks")
    return build_chroma_from_documents(split_docs, chroma_dir, embed_model)


def get_splitter(splitter_type: str):
    SPLIT_CHUNK_LINES = 10
    SPLIT_OVERLAP = 5

    if splitter_type == "code":
        PY_LANGUAGE = Language(tspython.language())
        PARSER = Parser(PY_LANGUAGE)

        print("[_get_splitter] using code splitter")

        return CodeSplitter(
            language="python",
            parser=PARSER,
            chunk_lines=SPLIT_CHUNK_LINES,
            chunk_lines_overlap=SPLIT_OVERLAP,
            include_metadata=True,
        )
    elif splitter_type == "text":
        print("[_get_splitter] using text splitter")
        return RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")
