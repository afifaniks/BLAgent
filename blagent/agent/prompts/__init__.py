RANKER_AGENT_PROMPT = """
You are a powerful AI code assistant.

BUG REPORT:
\"\"\"
{problem_statement}
\"\"\"

You are provided with a list of candidate code files that may be relevant to this bug.

Your task:
1. Use the `ReadFileSkeleton` tool to inspect the structure of each file.
2. Assign a relevance score from 0 (not relevant) to 10 (definitely needs to be modified) to each file.
   The score should reflect how likely the file is to require changes to fix the bug described.
3. You CAN view the file skeletons before assigning scores.
4. Construct a dictionary mapping file paths to their relevance scores.
5. You don't have to view all files if you already find the actual file. Stop once you are confident about the actual file.
6. You should stop as soon as you can.
7. YOU MUST provide score for 10 files but don't need to view all skeleton unless you absolutely must.

MUST NOTE:
- START with the most relevant files based on your initial understanding of the bug report
- MULTIPLE FILES CAN HAVE the SAME SCORE.
- SHOULD PROVIDE the complete file paths as they appear for the `ReadFileSkeleton` tool.
- The provided file paths are sorted by the chunk similarity scores but AVOID making assumptions based on order.
- Strictly return the list in JSON format.

IMPORTANT STOP CONDITION:
- After you come to conclusion with the scores, you must immediately output the Final Answer in the required JSON format and STOP.
- Do NOT write another Thought: after receiving the final Observation.
- The very last thing in your output must be:

Final Answer:
```json
{{ ... }}
```
FORMAT INSTRUCTIONS:

You must respond using this ReAct format exactly:

Thought: I need to inspect the structure of fileA.py
Action: ReadFileSkeleton
Action Input: "fileA.py"

Observation: [skeleton output here]

Final Answer:
```json
{{
    "fileB.py": 8,
    "fileA.py": 7,
    "fileC.py": 7
}}
Your candidate files:
{retrieved_file_paths}
"""

CREWAI_RANKER_AGENT_PROMPT = """
BUG REPORT:
\"\"\"
{problem_statement}
\"\"\"

Candidate files to analyze and rank:
{retrieved_file_paths}

Your primary goal is to accurately rank a list of candidate code files based on their relevance to a provided bug report.

Follow these steps to achieve your goal:
1. Systematically inspect each file from the candidate list by using the `ReadFileSkeleton` tool. This tool will provide you with the file's class names, function signatures, and docstrings.
2. Carefully analyze the `ReadFileSkeleton` output for each file in the context of the bug report.
3. Assign a relevance score from 0 to 10 to each file.
   - **10**: The file almost certainly contains code that needs to be modified to fix the bug.
   - **0**: The file is completely irrelevant.
   - **Intermediate scores**: Reflect partial relevance or a lower probability of being the correct fix location.
4. Your final output must be a valid JSON dictionary that maps each candidate file path to its relevance score. Ensure you use the exact file paths provided.

Final Output Format:
```json
{{
    "path/to/file1.py": 9,
    "path/to/file2.py": 4,
    "path/to/file3.py": 1
}}
Remember, your final answer should be only the JSON object, with no extra text or explanations.
"""

EVIDENCE_ANCHORED_RERANKER_PROMPT = """The following are the top {num_file_paths} ranked files retrieved for a given bug report.
Your job is to analyze the files and rerank them based on their relevance to the problem statement.
The idea is to find the actual file where a patch needs to be applied to fix the bug.

You should return a ranked list of file paths, ordered from most relevant to least relevant,
based on their content and relevance to the problem statement.
Example Output:
```json
{{
   "ranked_files": [
      "path/to/most_relevant_file.py",
      "path/to/second_most_relevant_file.py",
      ...
      "path/to/least_relevant_file.py"
   ]
}}
```
Do not include any explanations or additional text outside the JSON structure.

Problem Statement:
{problem_statement}
Possibly Relevant Files:
{aggregated_code_text}
"""

FUNCTION_LEVEL_LOCALIZATION_PROMPT = """
The following are the top {num_file_paths} ranked files retrieved for a given bug report.
Your job is to analyze the python files and method to rerank them based on their relevance to the problem statement.
The idea is to find the actual python method where a patch needs to be applied to fix the bug.

You should return a ranked list of file paths and Class::method names, ordered from most relevant to least relevant,
based on their content and relevance to the problem statement. Think and analyze properly to return the Class::function/method name.
If there is no class, just return the function name. If there is no function, just return the class name.
If there is both, return in Class::method format.
Do not return more than 3 methods per file. You should return at least 10 Class/method names across all files.
Example Output:
```json
{{
   "ranked_files": [
      {{"path/to/most_relevant_file.py": ["Class::most_relevant_method", "second_most_relevant_method"]}},
      {{"path/to/second_most_relevant_file.py": ["second_most_relevant_method"]}},
      ...
      {{"path/to/least_relevant_file.py": ["least_relevant_method"]}}
   ]
}}
```
Do not include any explanations or additional text outside the JSON structure.

Problem Statement:
{problem_statement}
Possibly Relevant Files:
{aggregated_code_text}
"""
