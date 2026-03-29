import re

SYSTEM_PROMPT = (
    "You are an AI assistant that rewrites Python bug reports and behavior descriptions into focused search queries "
    "for retrieving the most relevant source code responsible for the issue.\n\n"
    "Focus on the core of the issue by including:\n"
    "- The exact behavior being reported\n"
    "- Concrete technical anchors: model/field names, imports, functions, decorators, traceback entries, CLI commands, config keys, or version info\n"
    "- Any code paths, imports that could be relevant to the issue.\n"
    "- The probable reason of the issue\n"
    "- The expected vs. actual result (e.g., validation ignored, incorrect path, wrong class scope)\n"
    "- Any reproduction details that help narrow the source logic\n"
    "- Avoid generic summaries; prefer code-level precision and phrasing tied to symptoms or error causes\n\n"
    "Example:\n"
    """bug in print_changed_only in new repr: vector values
    ```python
    import sklearn
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    sklearn.set_config(print_changed_only=True)
    print(LogisticRegressionCV(Cs=np.array([0.1, 1])))
    ```
    > ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    Augmented:
    The error occurs in `LogisticRegressionCV` when `print_changed_only=True` and `Cs` is a numpy array, raising a ValueError about ambiguous array truth value during parameter comparison. The issue likely stems from a boolean check in the repr logic comparing `Cs` (array) to its default value (`np.logspace(-4, 4, 8)`), where element-wise equality produces an array of booleans instead of a scalar. The problematic code likely resides in `__init__` or parameter validation logic using `if Cs != default`, requiring `.all()` or `.any()` for array comparisons. The error is triggered during `print()` due to `set_config(print_changed_only=True)` modifying the repr behavior.
    """
    "Output only the rewritten query text. No sections, headers, or markdowns. Be retrieval-friendly and concise, with no extra text. Aim for 2-4 sentences."
)

BEHAVIORAL_SYSTEM_PROMPT = (
    "You are an AI assistant that rewrites Python bug reports and behavioral descriptions into focused search queries "
    "for retrieving the most relevant source code responsible for the issue.\n\n"
    "Focus on the user-observed behavior and triggering conditions. Include:\n"
    "- The exact observed behavior (e.g., silent failure, incorrect output, crash)\n"
    "- Expected vs. actual behavior\n"
    "- Triggers such as input parameters, environment, CLI flags, or API calls\n"
    "- Probable module or component involved (if clearly inferable)\n"
    "- Possible reason behind the behavior (e.g., missing fallback, incorrect state check)\n"
    "- Avoid speculative deep internals; keep it grounded in externally visible symptoms\n\n"
    "Example:\n"
    """Bug:
    When training a model using `transformers.Trainer` with `fp16=True` on a 4GB GPU, training silently hangs. No traceback or error message is shown; the training loop remains stuck after the first step.

    Augmented:
    `transformers.Trainer` hangs during training when `fp16=True` and VRAM is low (e.g., 4GB GPU). Likely module involved: `accelerate` or mixed precision handling in `Trainer`. Expected OOM error or graceful fallback, but instead the training loop freezes without exception, possibly due to unhandled CUDA error or silent failure in gradient scaler."""
    "Output only the rewritten query text. No sections, headers, or markdowns. Be retrieval-friendly and concise, with no extra text. Aim for 2-4 sentences."
)

STRUCTURAL_SYSTEM_PROMPT = (
    "You are an AI assistant that rewrites Python bug reports and technical descriptions into focused search queries "
    "to retrieve source code most relevant to the root cause of the issue.\n\n"
    "Focus on the code structure and error source. Include:\n"
    "- Relevant functions, methods, or classes (e.g., `__init__`, `from_pretrained`, `__repr__`)\n"
    "- Module or package names (e.g., `sklearn.linear_model`, `torch.nn.functional`)\n"
    "- File or traceback context if available (e.g., error in `sklearn/utils/validation.py`)\n"
    "- Probable reason (e.g., wrong type comparison, improper shape validation, missing null check)\n"
    "- Avoid restating full tracebacks or irrelevant details\n"
    "- Do not include user-defined variable names, class names, or specific instance details unless they are part of the core issue.\n\n"
    "Example:\n"
    """Bug:
    With `sklearn.set_config(print_changed_only=True)`, printing `LogisticRegressionCV(Cs=np.array([0.1, 1]))` raises:
    ValueError: The truth value of an array with more than one element is ambiguous.

    Augmented:
    In `sklearn.linear_model.LogisticRegressionCV`, enabling `print_changed_only=True` causes a ValueError when `Cs` is a numpy array. The issue likely stems from a parameter comparison in `__repr__` or `__init__` using `!=` on arrays without `.all()` or `.any()`. Suspected faulty logic in parameter diffing or config-aware repr code in `sklearn.utils._param_validation` or related helpers."""
    "Output only the rewritten query text. No sections, headers, or markdowns. Be retrieval-friendly and concise, with no extra text. Aim for 2-4 sentences."
)


# self.system_prompt = (
#     "You are an AI assistant that rewrites technical bug reports into focused, code-aware search queries. "
#     "Your rewritten query should help locate the relevant source code responsible for the observed issue.\n\n"
#     "Focus on the core of the issue by including:\n"
#     "- The **precise behavior or failure** (e.g., crash, wrong output, missing fallback, recursion error)\n"
#     "- Any **specific functions, classes, modules, or symbols** mentioned (e.g., `ccode`, `sinc`, `sympify`, `SCRIPT_NAME`, `is_zero`)\n"
#     "- **Expected vs. actual behavior**, especially when fallback logic is missing or the wrong code path is triggered\n"
#     "- Highlight if the error caused by any operators like `+`, `-`, `*`, `/`, `**`\n"
#     "- Any **version-specific compatibility problems**, such as serialization or platform differences (e.g., Python 2→3, different OSes)\n"
#     "- Highlight if it's a code generation, parsing, symbolic math, config handling, or cross-version issue\n\n"
#     "Do not summarize or restate the problem verbatim. Instead, synthesize a **concise, retrieval-optimized query** that captures what went wrong and where. "
#     "Be clear, technically specific, and avoid generalities. Your output should be 2-4 sentences, with no extra explanation."
# )


class QueryAugmentor:
    """
    A utility class to rewrite verbose natural language queries into
    concise, code-retrieval-optimized queries using an Ollama model.
    """

    def __init__(self, ollama_client, model_name: str = "llama3"):
        """
        Initialize the query augmentor.

        Args:
            ollama_client (OllamaClient): An instance of OllamaClient.
            model_name (str): The model to use for rewriting queries.
        """
        self.client = ollama_client
        self.model_name = model_name
        # self.system_prompt = SYSTEM_PROMPT

    def augment(self, query: str, temperature: float = 0, type: int = 0) -> str:
        """
        Rewrite a verbose natural language query into a retrieval-optimized query.

        Args:
            query (str): The original problem statement.

        Returns:
            str: The rewritten query optimized for retrieval.
        """
        if type == 0:
            system_prompt = STRUCTURAL_SYSTEM_PROMPT
        elif type == 1:
            system_prompt = BEHAVIORAL_SYSTEM_PROMPT
        else:
            raise ValueError(
                "Invalid type specified. Use 0 for structural, 1 for behavioral."
            )

        response = self.client.chat_completion(
            model_name=self.model_name,
            prompt=query,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        try:
            content = response["content"]
        except KeyError:
            content = response["message"]["content"]
            
        content = re.sub(r"<think>(.*?)</think>", "", content, flags=re.DOTALL)

        return content.strip()


# import re

# from openai import OpenAI


# class QueryAugmentor:
#     """
#     A utility class to rewrite verbose natural language queries into
#     concise, code-retrieval-optimized queries using the OpenAI API.
#     """

#     def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
#         """
#         Initialize the query augmentor.

#         Args:
#             model_name (str): The OpenAI model to use (e.g., "gpt-4o").
#             api_key (str): Optional API key, otherwise uses environment variable.
#         """
#         self.model_name = model_name
#         self.client = OpenAI(
#             api_key=""
#         )
#         self.system_prompt = SYSTEM_PROMPT

#     def augment(self, query: str) -> str:
#         """
#         Rewrite a verbose natural language query into a retrieval-optimized query.

#         Args:
#             query (str): The original problem statement.

#         Returns:
#             str: The rewritten query optimized for retrieval.
#         """
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {"role": "system", "content": self.system_prompt},
#                 {"role": "user", "content": query},
#             ],
#             temperature=0,
#         )
#         content = response.choices[0].message.content
#         return content.strip()
