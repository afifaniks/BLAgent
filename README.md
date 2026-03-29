# BLAgent: Agentic RAG for File-Level Bug Localization


## Prerequisites
This repository uses Ollama-based LLMs. Make sure to have Ollama up and running following: https://docs.ollama.com/quickstart

Once Ollama is installed. Pull the LLMs from Ollama library. In our experiments, we use:

`gpt-oss-120b`: https://ollama.com/library/gpt-oss:120b

`qwen3-32b`: https://ollama.com/library/qwen3

These models are required to reproduce the results.

## Setup

Clone the repository and go to the directory
```bash
cd BLAgent
```
Setup environment
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OLLAMA_HOST=http://localhost:11434
```
Install dependencies
```bash
pip install -r requirements.txt
```

### APR Setup
To setup Agentless, you can either follow their official documentation where they use proprietary LLM like GPT-4o: https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md

Or, if you want to use it with Ollama, you can use our ported version of Agentless. The instruction is provided later in this documentation.

## Preprocessing
For faster processing, we process SWE-bench-Lite dataset into json files. This can be done by running:

```bash
python preprocess/process_swe_dataset.py
```
It will process the repositories under `swebench_data` directory in the project root. It will create two json files for each repo.  

For example
```bash
swebench_data/swebench_0_raw.json # Containing all file_path and file contents for repository at index 0 of the dataset
swebench_data/swebench_0_py.json # Containing only python file (.py) paths and file contents for repository at index 0 of the dataset
```
Each repository is identified by the number of index in the dataset.

## Running the pipeline
To run the RAG pipeline on swebench dataset in different configuration:

Create a directory to store the results
```bash
mkdir retrieval_results
```

### Dense Retrieval

#### Dense Retrieval with Naive Text Chunking
```bash
python pipelines/rag_pipeline.py --splitter text --results_path retrieval_results/dense_retrieval_text_results.json --chroma_root chroma_data_text
```

#### Dense Retrieval with Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/dense_retrieval_code_results.json --chroma_root chroma_data_code
```

#### Dense Retrieval with Path-aware Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/dense_retrieval_code_with_path_results.json --chroma_root chroma_data_code_with_path --include_path
``` 

### Basic RAG Pipeline

#### Basic RAG Pipeline with Naive Text Chunking
```bash
python pipelines/rag_pipeline.py --splitter text --results_path retrieval_results/rag_text_results.json --chroma_root chroma_data_text --use_rag --llm gpt-oss:120b
```
#### Basic RAG Pipeline with Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/rag_code_results.json --chroma_root chroma_data_code --use_rag --llm gpt-oss:120b
```

#### Basic RAG Pipeline with Path-aware Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/rag_code_with_path_results.json --chroma_root chroma_data_code_with_path --include_path --use_rag  --llm gpt-oss:120b
```


## BLAgent Pipeline

Since the complete Agentic Pipeline (BLAgent) uses query transformation, it is suggested that first query transformation pipeline is executed. Once it is executed we can simply use the output of the query transformation candidates as input.

### Query Transformation
```bash
python pipelines/query_transformation.py --results_path retrieval_results/query_transformation_results.json --chroma_root chroma_data_code_with_path --llm gpt-oss:120b
```
### Agentic Reranking

The agentic reranking stage is composed of 2 phases. Each phase generates file-level localization results that can be used for downstream localization or repair. In Phase 1, the ReAct agent goes over the candidate files from the retrievers and analyzes the code skeletons (e.g., function signatures, code skeletons, docstrings) and generates a ranked list of files. In Phase 2, a single inference is done over the Top-5 files over a pruned code context to further refine the rankings (i.e., reranking).

#### Phase 1: Skeleton-Based Agent Scoring (SAS)

```bash
python pipelines/agentic_pipeline.py --input_file retrieval_results/query_transformation_results.json --results_path retrieval_results/agentic_ranked_results.json --llm gpt-oss:120b
```
The pipeline currently supports Ollama models, and Anthropic models. However, it should be easily extendable to other models as well. Some supported models:

```bash
claude-4-6-sonnet (Anthropic)
gpt-oss:120b (Ollama)
qwen3:32b (Ollama)
```

This pipeline generates a ranked list of file paths for each bug similar to the following:

```json
{
    "astropy/modeling/separable.py": 10,
    "astropy/modeling/core.py": 8,
    "astropy/modeling/models.py": 7,
    "astropy/modeling/mappings.py": 5,
    "astropy/modeling/functional_models.py": 3,
    "astropy/modeling/projections.py": 3,
    "astropy/io/misc/asdf/tags/transform/compound.py": 2,
    "astropy/modeling/bounding_box.py": 2,
    "astropy/modeling/polynomial.py": 1,
    "astropy/modeling/tabular.py": 1
}
```

#### Phase 2: Evidence-Anchored Reranking (EAR)
This is an optional but recommended step to further refine the agentic reranking on Phase 1. In our observation, if the Phase 1 is run with a powerful model like `Claude-4-6-Sonnet`, this step might not be necessary if API cost is a concern. However, when using smaller models like `GPT-OSS` or `Qwen3`, this phase can significantly improve overall performance. 


```bash
python blagent/agent/evidence_anchored_rerank.py --model gpt-oss:120b --max_files_to_rerank 5 --num_chunks_per_file 5 --chroma_root chroma_data_code_with_path --source retrieval_results/agentic_ranked_results.json --results_path retrieval_results/agentic_final_reranked_results.json
```

#### Function Level Localization
The Phase 2 (EAR) pipeline can also be used for function-level localization with a single parameter change. To run function level localization, run the following command:

```bash
python blagent/agent/evidence_anchored_rerank.py --model gpt-oss:120b --max_files_to_rerank 5 --num_chunks_per_file 5 --chroma_root chroma_data_code_with_path --source retrieval_results/agentic_ranked_results.json --results_path retrieval_results/function_level_loc_results.json --function_level
```

This will result in outputs like this for each instance (method, class::method, etc.):

```json
"final_reranked_files": [
    {
        "astropy/modeling/separable.py": [
            "_separable",
            "_cstack",
            "separability_matrix"
        ]
    },
    {
        "astropy/modeling/core.py": [
            "Model::_calculate_separability_matrix",
            "CompoundModel::__init__",
            "CompoundModel::evaluate"
        ]
    },
...
]

```


## Evaluation and Results
Results and evaluation data are organized under [results](results) directory. Each subdirectory contains artifacts that were used to report results for each RQ. For example:

[results/rq1-1](results/rq1-1/) contains the artifacts/results generated by LocAgent with Claude-4.6-Sonnet in [results/rq1-1/locagent/locagent_results_claude46sonnet/](results/rq1-1/locagent/locagent_results_claude46sonnet/) directory.
```
results
├── rq1-1
│   ├── blagent_claude-4-6.json
│   ├── blagent_gpt-oss.json
│   └── locagent
│       ├── locagent_claude-4-6_loc_outputs_normalized.jsonl
│       ├── locagent_results_claude35sonnet
│       │   └── loc_outputs.jsonl
│       └── locagent_results_claude46sonnet
│           ├── args.json
│           ├── localize.log
│           ├── loc_outputs.jsonl
│           └── loc_trajs.jsonl
├── rq1-2
│   ├── rag_code_chunking.json
│   ├── rag_file_path_augmented_code_chunking.json
│   └── rag_text_chunking.json
├── rq1-3
│   └── blagent_gpt-oss_phase1_and_phase2_base_retrieval.json
...
```

#### File-level localization evaluation
We provide an evaluation script that allows to quickly generate most of the file-level localization results reported in different RQs in the paper for BLAgent.

```bash
python evaluation/ranking_evaluation.py
```
Example output:

*RF = Retrieval Failure
==================== ALL RESULTS ====================
| RQ      | Setting                                                     |    MRR |   Acc@1 |   Acc@3 |   Acc@5 |   Acc@10 | RF / Top10 Misses   |
|---------|-------------------------------------------------------------|--------|---------|---------|---------|----------|---------------------|
| RQ1.1   | BLAgent (GPT-OSS 120B)                                      | 0.8514 |  0.7867 |  0.9233 |  0.9333 |   0.9433 | 14/17               |
| RQ1.1   | BLAgent (Claude 4.6 Sonnet)                                 | 0.9009 |  0.8667 |  0.93   |  0.9467 |   0.9533 | 14/14               |
| RQ1.2   | RAG - Text Chunking                                         | 0.6904 |  0.6433 |  0.74   |  0.7467 |   0.75   | 75/75               |
| RQ1.2   | RAG - Code Aware Chunking                                   | 0.722  |  0.6667 |  0.7833 |  0.7933 |   0.8    | 60/60               |
| RQ1.2   | RAG - File Path Aware Code Chunking                         | 0.7347 |  0.6733 |  0.7967 |  0.82   |   0.8233 | 53/53               |
| RQ1.3   | BLAgent (GPT-OSS 120B) - Base Retrieval (Phase 1)           | 0.7697 |  0.6856 |  0.8395 |  0.8863 |   0.8963 | 31/31               |
| RQ1.3   | BLAgent (GPT-OSS 120B) - Base Retrieval (Phase 1 + Phase 2) | 0.8192 |  0.7625 |  0.8896 |  0.8896 |   0.8997 | 30/30               |
| RQ1.4   | BLAgent (Phase 1) T0 (GPT-OSS 120B)                         | 0.7856 |  0.6967 |  0.8567 |  0.9033 |   0.93   | 14/21               |
| RQ1.4   | BLAgent (Phase 1) T1 (GPT-OSS 120B)                         | 0.7806 |  0.6933 |  0.86   |  0.9    |   0.91   | 14/27               |
| RQ1.4   | BLAgent (Phase 1) T0 + T1 (GPT-OSS 120B) - T0 First         | 0.7946 |  0.71   |  0.86   |  0.9033 |   0.9433 | 14/17               |
| RQ1.4   | BLAgent (Phase 1) T0 + T1 (GPT-OSS 120B) - T1 First         | 0.7972 |  0.7167 |  0.8667 |  0.9033 |   0.93   | 14/21               |
| RQ1.5   | BLAgent (Claude 4.6) - Phase 1                              | 0.8896 |  0.8433 |  0.9333 |  0.9467 |   0.9533 | 14/14               |
| RQ1.5   | BLAgent (Claude 4.6) - Phase 1 + Phase 2                    | 0.9009 |  0.8667 |  0.93   |  0.9467 |   0.9533 | 14/14               |
| RQ1.5   | BLAgent (Qwen3 32B) - Phase 1                               | 0.7894 |  0.7067 |  0.8633 |  0.9    |   0.93   | 15/21               |
| RQ1.5   | BLAgent (Qwen3 32B) - Phase 1 + Phase 2                     | 0.8474 |  0.7967 |  0.8867 |  0.9033 |   0.9333 | 14/20               |
| RQ1.5   | BLAgent (GPT-OSS 120B) - Phase 1                            | 0.7946 |  0.71   |  0.86   |  0.9033 |   0.9433 | 14/17               |
| RQ1.5   | BLAgent (GPT-OSS 120B) - Phase 1 + Phase 2                  | 0.8514 |  0.7867 |  0.9233 |  0.9333 |   0.9433 | 14/17               |
| Dis 5.5 | Pruned Context 5 files 5 chunks (GPT-OSS)                   | 0.8514 |  0.7867 |  0.9233 |  0.9333 |   0.9433 | 14/17               |
| Dis 5.5 | Pruned Context 5 files 10 chunks (GPT-OSS)                  | 0.8273 |  0.7567 |  0.8967 |  0.9033 |   0.9433 | 14/17               |
| Dis 5.5 | Pruned Context 10 files 5 chunks (GPT-OSS)                  | 0.8441 |  0.7733 |  0.91   |  0.9267 |   0.9433 | 14/17               |
| Dis 5.5 | Pruned Context 10 files 10 chunks (GPT-OSS)                 | 0.851  |  0.78   |  0.93   |  0.9367 |   0.9433 | 14/17               |

Similarly, evaluation can be ran for new results by updating the ranking_evaluation.py file pointing to the new report file:

```json
{
    "rq": "X",
    "name": "New Agentic RAG",
    "path": "localization_results/new_agentic_ranked_results.json",
    "pred_list_name": "ranked_scores", # Should point to the accurate field name depending on retrieval technique (e.g., ranked_scores, rag_ranked_files, retrieved_files, etc.)
    "ground_truth": "patch_file", # Should be either "patch" or "patch_file"
},
```

#### Function-level localization evaluation

To evaluate function-level localization accuracy, please use this notebook [evaluation/function_eval.ipynb](evaluation/function_eval.ipynb)

Running all the cells should provide the Top-5 and Top-10 function-level localization accuracy:

```
Top-K Accuracy:
K     Accuracy     Correct/Total
----------------------------------------
Top-5   0.7810 (78.10%)  214/274
Top-10  0.8102 (81.02%)  222/274
```


# Repair with Agentless
First, we have to convert the BLAgent file-level localization output to suitable Agentless input file in `jsonl` format. For example, if we want to use our previous result file [localization_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json](localization_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json), we can do this:

```bash
python preprocess/convert_agentless.py --blagent_output retrieval_results/agentic_final_reranked_results.json --converted_output blagent_locs.jsonl
```

This will produce `blagent_locs.jsonl` file which is compatible to use in Agentless.

From this stage on https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md, we can follow the Agentless documentation to run the APR specifically from [2. localize to related elements](https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md#2-localize-to-related-elements) stage. However, since our ported Agentless uses Ollama as a backend, we just have to change the LLM and backend whenever an LLM is required to complete an agentless stage.

For example, to localize at the function level using `blagent_locs.jsonl` and with the Ollama-ported Agentless, we can use this:

```bash
python agentless/fl/localize.py --related_level \
                                --output_folder results/swe-bench-lite/related_elements_gpt_oss \
                                --top_n 3 \
                                --compress_assign \
                                --compress \
                                --start_file {DIRECTORY}/blagent_locs.jsonl \                                
                                --num_threads 10 \
                                --model gpt-oss:120b \
                                --backend ollama \
```                                

The last two parameters `(--model and --backend)` are particularly important to enable ollama, otherwise the framework will use its default LLM GPT-4o.

Similarly, for all subsequent steps should follow the official Agentless documentation along with the mentioned parameters.

## Patch Evaluation
Once patches are generated, they can be evaluated using swebench library. We recommend creating a new environment with [swebench 4+ (latest)](https://pypi.org/project/swebench/) as swebench 2.1 used by Agentless is now outdated and we ran into many problems to evaluate with it. The run command is simple:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path results/rq2-1/repair_results/run_3/blagent/patches/blagent_regr_repr_preds_main_gpt_oss.jsonl \
    --max_workers 10 \
    --run_id blagent_evaluation
```
Due to the limit to github file size, we provide only one final generated patches on swebench using BLAgent localization here: [results/rq2-1/repair_results/run_3/blagent/patches/blagent_regr_repr_preds_main_gpt_oss.jsonl](results/rq2-1/repair_results/run_3/blagent/patches/blagent_regr_repr_preds_main_gpt_oss.jsonl). 

However, we also provide all artifacts generated during Run 3. All the intermediate files/outputs generated while using both Agentless native localization and BLAgent while using Agentless APR can be downloaded from here: https://zenodo.org/records/19320256


## Additional Materials

We provide additional scripts and notebooks under the [`evaluation/`](evaluation/) directory to support deeper analysis and reproducibility.

### Repair Analysis

* [`evaluation/unique_repair_analysis.py`](evaluation/unique_repair_analysis.py):
  Analyzes unique repairs produced by each method and identifies cases where one method fails due to incorrect localization.

* [`evaluation/analyze_incorrect_patch_with_correct_loc.ipynb`](evaluation/analyze_incorrect_patch_with_correct_loc.ipynb):
  Investigates failure stages in APR (e.g., line-level localization vs. patch generation) when localization is correct but repair fails.

### Localization Evaluation

* [`evaluation/ranking_eval_jsonl.py`](evaluation/ranking_eval_jsonl.py):
  Evaluates file-level localization outputs in `.jsonl` format (e.g., LocAgent/Agentless outputs).

### Context and Efficiency Analysis

* [`evaluation/context_size_evaluation.py`](evaluation/context_size_evaluation.py):
  Examines how prompt size (number of files and chunks) impacts token usage and performance.

* [`evaluation/cost_analysis.ipynb`](evaluation/cost_analysis.ipynb):
  Provides cost analysis based on token usage and model inference.

### Stochasticity and Variability

* [`evaluation/analyze_repair_overlaps.ipynb`](evaluation/analyze_repair_overlaps.ipynb):
  Analyzes stochasticity and repair overlaps across multiple runs.


## Acknowledgements
We thank [Agentless](https://github.com/OpenAutoCoder/Agentless/tree/main) and [CoSIL](https://github.com/ZhonghaoJiang/CoSIL/tree/master) for their work and making it public for others to use.


