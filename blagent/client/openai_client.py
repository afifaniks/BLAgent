from typing import Optional

from openai import OpenAI


class OpenAIClient:
    def __init__(self):
        self.client = OpenAI()
        print(f"OpenAI client initialized")

    def chat_completion(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate a chat completion using OpenAI API.

        Args:
            model_name: Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response

        Returns:
            dict with 'content' and 'usage' keys
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        print(f"Calling OpenAI model '{model_name}' with temperature {temperature}...")

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as e:
            raise RuntimeError(f"Error during OpenAI chat completion: {e}")
