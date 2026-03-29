import asyncio
from typing import AsyncGenerator, Dict, List, Optional

import ollama
from ollama import ChatResponse, Client


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.client = Client(host=base_url)
        print(f"Ollama base URL set to {base_url}")

    def list_models(self) -> List[str]:
        """
        List all available models.
        Returns a list of model names.
        """
        try:
            models = ollama.list().get("models", [])
            return [model.model for model in models]
        except Exception as e:
            raise RuntimeError(f"Error fetching model list: {e}")

    def model_exists(self, model_name: str) -> bool:
        try:
            models = ollama.list().get("models", [])
            existing_model_names = [model.model for model in models]
            if model_name in existing_model_names:
                return True
            else:
                print(
                    f"Model '{model_name}' does not exist. Available models: {existing_model_names}"
                )
                return False
        except Exception as e:
            print(f"Error checking model list: {e}")
            return False

    def pull_model(self, model_name: str, insecure: bool = False) -> None:
        if not self.model_exists(model_name):
            try:
                print(f"Pulling model '{model_name}'...")
                self.client.pull(model=model_name, insecure=insecure)
            except Exception as e:
                raise RuntimeError(f"Could not pull model '{model_name}': {e}")
        else:
            print(f"Model '{model_name}' already exists.")

    def chat_completion(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> ChatResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        print(f"Calling model '{model_name}' with temperature {temperature}...")
        return self.client.chat(
            model=model_name, messages=messages, options={"temperature": temperature}
        )

    def embed(self, model_name: str, text: str) -> List[float]:
        try:
            response = self.client.embed(model=model_name, input=text)
            return response["embedding"]
        except Exception as e:
            raise RuntimeError(f"Error during embedding: {e}")

    async def chat_completion_async(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously stream chat completion responses, yielding tokens one by one.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        loop = asyncio.get_event_loop()

        def sync_chat_stream():
            return ollama.chat(model=model_name, messages=messages, stream=True)

        try:
            # Run blocking chat stream in a thread
            stream = await asyncio.to_thread(sync_chat_stream)
            for chunk in stream:
                content = chunk["message"]["content"]
                yield content
        except Exception as e:
            raise RuntimeError(f"Error during chat streaming: {e}")
