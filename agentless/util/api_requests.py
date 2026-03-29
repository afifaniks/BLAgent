import time
from typing import Dict, Union

import anthropic
import openai
import tiktoken
import time
from typing import Dict, Optional
from typing import Dict
import ollama


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret


def create_ollama_config(
    message: str,
    max_tokens: int,
    temperature: float = 0.8,
    model: str = "llama3",
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    """
    Creates a configuration dictionary for the Ollama API.
    """
    if isinstance(message, list):
        # Convert list of messages to Ollama format
        # Ollama expects a list of dictionaries with 'role' and 'content' keys
        messages = message
        # Check for system message and add if not present
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_message})
    else:
        # Create messages for single string input
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]

    config = {
        "model": model,
        "messages": messages,
        "options": {
            "num_predict": 8000,  # Ollama uses num_predict for max tokens
            "temperature": temperature,
        },
    }
    return config


def request_ollama_engine(config, logger, max_retries=40, timeout=100):
    """
    Handles API requests to the Ollama engine.
    """
    ret = None
    retries = 0

    while ret is None and retries < max_retries:
        try:
            logger.info("Creating Ollama API request")

            # The ollama library's chat function is a good match for the existing handler's structure.
            # We use a stream=False to get the full response at once.
            ret = ollama.chat(**config, stream=False)

            # Ollama's response format is a dictionary. We extract the message content.
            if ret and "message" in ret and "content" in ret["message"]:
                logger.info(f"Ollama API response: {ret['message']['content']}")
            else:
                logger.error("Unexpected Ollama API response format.")
                ret = None
                time.sleep(1)  # Wait before retrying

        except Exception as e:
            logger.error(f"An unexpected error occurred while calling Ollama API: {e}", exc_info=True)
            time.sleep(1)  # Wait for a few seconds before retrying

        retries += 1

    return ret


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    # Define a user prompt to send to the model.
    user_prompt = "What is the capital of France?"
    max_response_tokens = 50

    print(f"Creating a config for the Ollama model with prompt: '{user_prompt}'")

    # Create the configuration dictionary for the Ollama request.
    ollama_config = create_ollama_config(
        message=user_prompt,
        max_tokens=max_response_tokens,
        model="qwen3:32b",  # Ensure you have this model pulled locally
    )

    print("Requesting response from the Ollama engine...")

    # Send the request to the Ollama engine.
    response = request_ollama_engine(config=ollama_config, logger=logger)

    # Check if a valid response was received and print the content.
    if response and "message" in response and "content" in response["message"]:
        print("\n--- Ollama Response ---")
        print(response["message"]["content"])
        print("-----------------------")
        print(response)
    else:
        print("\nFailed to get a valid response from the Ollama engine.")
        print(
            "Please ensure that Ollama is running and the specified model is available."
        )
