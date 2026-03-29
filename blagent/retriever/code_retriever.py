import json
import os
from collections import defaultdict
from pprint import pprint

import tree_sitter_python as tspython
from datasets import load_dataset
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import CodeSplitter
from tree_sitter import Language, Parser

from blagent.git.repository import get_project_structure_from_scratch
from blagent.util.preprocess_data import filter_none_python, filter_out_test_files


class CodeRetriever:
    def __init__(
        self,
        data_dir,
        repo_root,
        chroma_base_dir,
        model_name="nomic-ai/nomic-embed-text-v1",
        chunk_lines=35,
        chunk_overlap=5,
        device="cuda",
    ):
        self.data_dir = data_dir
        self.repo_root = repo_root
        self.chroma_base_dir = chroma_base_dir

        # Set up Tree-sitter parser
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

        # Set up code splitter
        self.splitter = CodeSplitter(
            language="python",
            parser=self.parser,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_overlap,
            include_metadata=True,
        )

        # Embedding model
        self.embed_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": False},
            show_progress=True,
        )

    def _collect_code_nodes(self, node, path_parts, repo_name):
        documents = []
        relative_path = os.path.join(*path_parts)
        if relative_path.endswith(".py") and isinstance(node, dict):
            try:
                code = "\n".join(node["text"])
                metadata = {
                    "repo": repo_name,
                    "file": relative_path,
                    "relative_path": os.path.join(repo_name, relative_path),
                    "type": "file",
                }
                documents.append(Document(page_content=code, metadata=metadata))
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")
        elif isinstance(node, dict):
            for key, subnode in node.items():
                documents.extend(
                    self._collect_code_nodes(subnode, path_parts + [key], repo_name)
                )
        return documents

    def _load_documents_from_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        all_documents = []
        for repo_name, repo_content in data.items():
            all_documents.extend(
                self._collect_code_nodes(repo_content, [""], repo_name)
            )
        return all_documents

    def _prepare_repository(self, swe_data_index, swebench):
        swe_data = swebench[swe_data_index]
        repo_json_path = os.path.join(
            self.data_dir, f"swebench_{swe_data_index}_py.json"
        )
        chroma_dir = os.path.join(
            self.chroma_base_dir, f"chroma_repo_index_swe_data{swe_data_index}"
        )

        if not os.path.exists(repo_json_path):
            structure = get_project_structure_from_scratch(
                swe_data["repo"],
                swe_data["base_commit"],
                swe_data["instance_id"],
                self.repo_root,
                cleanup=True,
            )["structure"]
            filter_none_python(structure)
            filter_out_test_files(structure)
            with open(repo_json_path, "w") as f:
                json.dump(structure, f, indent=2)

        docs = self._load_documents_from_json(repo_json_path)

        if os.path.exists(chroma_dir):
            vector_store = Chroma(
                embedding_function=self.embed_model, persist_directory=chroma_dir
            )
        else:
            split_docs = []
            for doc in docs:
                try:
                    chunks = self.splitter.split_text(doc.page_content)
                    file_path = doc.metadata.get("relative_path", "")
                    for idx, chunk in enumerate(chunks):
                        split_docs.append(
                            Document(
                                page_content=f"[PATH] {file_path}\n[CODE]\n{chunk}",
                                metadata={**doc.metadata, "split_index": idx},
                            )
                        )
                except Exception as e:
                    print(
                        f"Failed to split {doc.metadata.get('relative_path', '')}: {e}"
                    )

            vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embed_model,
                persist_directory=chroma_dir,
            )
            vector_store.persist()

        return vector_store

    def retrieve_similar_code(
        self, swe_data_index, swebench, k=100, augmented_query=None
    ):
        swe_data = swebench[swe_data_index]
        chunk_map = {}
        vector_store = self._prepare_repository(swe_data_index, swebench)

        query = augmented_query or swe_data["problem_statement"]
        # Use similarity_search_with_score to get distance metrics
        # Note: Chroma returns DISTANCE (lower is better), not similarity
        results = vector_store.similarity_search_with_score(query, k=k * 10)

        file_distances = defaultdict(list)

        for doc, distance in results:
            path = doc.metadata["relative_path"]
            file_distances[path].append(distance)
            chunk_map.setdefault(path, [])
            seen_chunks = {entry["chunk"] for entry in chunk_map[path]}

            if doc.page_content not in seen_chunks:
                chunk_map[path].append(
                    {"chunk": doc.page_content, "distance": distance}
                )

        # Use the minimum (best) distance per file (lower is better)
        min_file_distances = {
            path: min(distances) for path, distances in file_distances.items()
        }
        # Sort by distance ascending (lower distance = better match)
        top_files = sorted(
            min_file_distances.items(), key=lambda x: x[1], reverse=False
        )[:k]
        unique_paths = [path for path, _ in top_files]

        return {
            "swe_data_index": swe_data_index,
            "problem_statement": swe_data["problem_statement"],
            "patch": swe_data["patch"],
            "retrieved_files": unique_paths,
            "file_chunks": {path: chunk_map[path][:5] for path in unique_paths},
        }
