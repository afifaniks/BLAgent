import json
import re

import tiktoken
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

from blagent.agent import prompts
from blagent.util.code_util import get_code_text_from_path
from blagent.util.signature_extractor import extract_signature_tree


class TokenCountingCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Count tokens in each prompt
        for prompt in prompts:
            tokens = len(tiktoken.get_encoding("cl100k_base").encode(prompt))
            self.prompt_tokens += tokens

    def on_llm_end(self, response, **kwargs):
        # Count tokens in each response
        for generation in response.generations:
            for gen in generation:
                tokens = len(tiktoken.get_encoding("cl100k_base").encode(gen.text))
                self.completion_tokens += tokens

        self.total_tokens = self.prompt_tokens + self.completion_tokens


class TokenCountingCallbackAPI(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        # Get actual token usage from OpenAI API response
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.prompt_tokens += token_usage.get("prompt_tokens", 0)
            self.completion_tokens += token_usage.get("completion_tokens", 0)
            self.total_tokens += token_usage.get("total_tokens", 0)


class SWEBenchRankerAgent:
    def __init__(
        self,
        repo_content: dict,
        retrieved_docs: dict,
        model_name: str = "qwen3:32b",
        temperature: float = 0.5,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.repo_content = repo_content
        self.retrieved_docs = retrieved_docs
        self.token_callback = None

        if model_name in ["gpt-4", "gpt-3.5-turbo", "claude-4-6-sonnet", "gpt-4o"]:
            self.token_callback = TokenCountingCallbackAPI()
        else:
            self.token_callback = TokenCountingCallback()

        # Tools — no sorting tool now, LLM just returns dict
        self.tools = [
            Tool(
                name="ReadFileSkeleton",
                func=self.read_file_skeleton_tool,
                description="Inspect the structure of a code file: class names, function signatures, docstrings.",
            ),
            # Tool(
            #     name="ViewMethod",
            #     func=self.read_method_body_tool,
            #     description="View the complete method implementation.",
            # ),
        ]

        # Initialize LLM
        if self.model_name in ["gpt-4", "gpt-4o", "gpt-5", "gpt-5.1", "gpt-3.5-turbo"]:
            print(f"Initializing OpenAI LLM: {self.model_name}")
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                callbacks=[self.token_callback],
            )

        elif "claude" in self.model_name.lower():
            print(f"Initializing Anthropic LLM: {self.model_name}")
            self.llm = ChatAnthropic(
                model=self.model_name,  # or claude-3-opus-20240229, claude-3-haiku-20240307
                temperature=self.temperature,
                callbacks=[self.token_callback],
            )

        else:
            print(f"Initializing Ollama LLM: {self.model_name}")
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                callbacks=[self.token_callback],
            )

        # Initialize Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"stop": ["\nThought:"]},
        )

    def read_file_skeleton_tool(self, file_path):
        file_path = file_path.replace('"', "").strip()
        print("CURRENT FILE:", file_path)
        code_text = get_code_text_from_path(self.repo_content, file_path.split("/"))
        return extract_signature_tree(code_text)

    # def read_method_body_tool(self, file_path, method_name):
    #     file_path = file_path.replace('"', "").strip()
    #     method_name = method_name.replace('"', "").strip()
    #     print("CURRENT FILE:", file_path)
    #     code_text = get_code_text_from_path(self.repo_content, file_path.split("/"))
    #     # Extract method body
    #     method_body = extract_method_body(code_text, method_name)
    #     return method_body if method_body else "Method not found."

    def _extract_final_json(self, output_text: str):
        """Extracts and parses the final JSON block from agent output."""
        match = re.search(r"```json\s*(\{.*?\})\s*```", output_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"(\{(?:.|\n)*\})", output_text)
            if not match:
                return None
            json_str = match.group(1)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*}", "}", json_str)
            repaired = re.sub(r",\s*\]", "]", repaired)
            try:
                return json.loads(repaired)
            except Exception:
                return None

    def sort_results(self, score_dict: dict, descending: bool = True):
        """Sorts a {file_path: score} dict into a sorted list of tuples."""
        return sorted(score_dict.items(), key=lambda x: x[1], reverse=descending)

    def run(self, max_files: int = 15):
        self.token_callback.prompt_tokens = 0
        self.token_callback.completion_tokens = 0
        self.token_callback.total_tokens = 0

        problem_statement = self.retrieved_docs["problem_statement"]
        seen = set()
        retrieved_file_paths = []
        for path in (
            self.retrieved_docs["retrieved_files_t0"][:max_files]
            + self.retrieved_docs["retrieved_files_t1"][:max_files]
        ):
            if path not in seen:
                seen.add(path)
                retrieved_file_paths.append(path)

        print("Total retrieved_files:", len(retrieved_file_paths))
        agent_prompt = prompts.RANKER_AGENT_PROMPT.format(
            problem_statement=problem_statement,
            retrieved_file_paths=retrieved_file_paths,
        )

        # Run the agent
        output_text = self.agent.run(agent_prompt)
        print(f"Prompt Tokens: {self.token_callback.prompt_tokens}")
        print(f"Completion Tokens: {self.token_callback.completion_tokens}")
        print(f"Total Tokens: {self.token_callback.total_tokens}")

        usage = {
            "prompt_tokens": self.token_callback.prompt_tokens,
            "completion_tokens": self.token_callback.completion_tokens,
            "total_tokens": self.token_callback.total_tokens,
        }

        parsed_output = self._extract_final_json(output_text)

        if parsed_output is not None:
            return parsed_output, usage
        else:
            print("⚠ Warning: Could not parse JSON. Returning raw output.")
            return output_text, usage
