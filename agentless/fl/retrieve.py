import argparse
import concurrent.futures
import json
import os
from threading import Lock

from datasets import load_dataset
from tqdm import tqdm

from agentless.fl.Index import EmbeddingIndex
from agentless.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
    get_repo_structure,
)
from agentless.util.utils import load_json, load_jsonl, setup_logger


def safe_retrieve_locs(bug, args, swe_bench_data, found_files, prev_o, write_lock=None):
    is_done = False
    max_retries = 5

    # Make a copy of args
    copy_args = argparse.Namespace(**vars(args))

    while not is_done and max_retries > 0:
        try:
            print(
                f"Attempting to process bug {bug['instance_id']} with chunk size {copy_args.chunk_size}"
            )
            retrieve_locs(
                bug, copy_args, swe_bench_data, found_files, prev_o, write_lock
            )
            is_done = True
        except Exception as e:
            print(f"Error processing bug {bug['instance_id']}: {e}")
            max_retries -= 1
            copy_args.chunk_size *= 2
            print(f"Doubling chunk size to {copy_args.chunk_size} and retrying...")


def retrieve_locs(bug, args, swe_bench_data, found_files, prev_o, write_lock=None):

    found = False
    for o in prev_o:
        if o["instance_id"] == bug["instance_id"]:
            found = True
            break

    if found:
        print(f"skipping {bug['instance_id']} since patch already generated")
        return None

    print(f"Processing bug {bug['instance_id']}")
    instance_id = bug["instance_id"]

    if args.target_id is not None:
        if args.target_id != instance_id:
            return None

    log_file = os.path.join(args.output_folder, "retrieval_logs", f"{instance_id}.log")
    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = get_repo_structure(
        instance_id, bug["repo"], bug["base_commit"], "playground"
    )

    filter_none_python(structure)
    filter_out_test_files(structure)

    if args.filter_file:
        kwargs = {  # build retrieval kwargs
            "given_files": [x for x in found_files if x["instance_id"] == instance_id][
                0
            ]["found_files"],
            "filter_top_n": args.filter_top_n,
        }
    else:
        kwargs = {}

    # main retrieval
    retriever = EmbeddingIndex(
        instance_id,
        structure,
        problem_statement,
        persist_dir=args.persist_dir,
        filter_type=args.filter_type,
        index_type=args.index_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        logger=logger,
        **kwargs,
    )

    file_names, meta_infos, traj = retriever.retrieve(mock=args.mock)

    print(file_names)

    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "found_files": file_names,
                    "node_info": meta_infos,
                    "traj": traj,
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()


def retrieve(args):
    if args.filter_file:
        found_files = load_jsonl(args.filter_file)
    else:
        found_files = []

    swe_bench_data = load_dataset(args.dataset, split="test")
    prev_o = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []
    print(f"Loaded {len(prev_o)} bugs from output file")

    if args.num_threads == 1:
        for bug in tqdm(swe_bench_data, colour="MAGENTA"):
            safe_retrieve_locs(
                bug, args, swe_bench_data, found_files, prev_o, write_lock=None
            )
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    safe_retrieve_locs,
                    bug,
                    args,
                    swe_bench_data,
                    found_files,
                    prev_o,
                    write_lock,
                )
                for bug in swe_bench_data
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(swe_bench_data),
                colour="MAGENTA",
            ):
                pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="retrieve_locs.jsonl")
    parser.add_argument(
        "--index_type", type=str, default="simple", choices=["simple", "complex"]
    )
    parser.add_argument(
        "--filter_type", type=str, default="none", choices=["none", "given_files"]
    )
    parser.add_argument("--filter_top_n", type=int, default=None)
    parser.add_argument("--filter_file", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--persist_dir", type=str)
    parser.add_argument("--target_id", type=str)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests (WARNING, embedding token counts are only accurate when thread=1)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
    )

    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    assert (
        not args.filter_type == "given_files" or args.filter_file != ""
    ), "Need to provide a filtering file"

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "retrieval_logs"), exist_ok=True)

    # dump argument
    with open(os.path.join(args.output_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    retrieve(args)


if __name__ == "__main__":
    main()
