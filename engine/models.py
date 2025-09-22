"""
Model routing abstraction for Curiosity AI.

This module defines a small wrapper around different language model providers.
It supports calling local models via [Ollama](https://ollama.com/), as well
as remote providers such as OpenAI and Anthropic.  The class exposes a
uniform asynchronous interface for generating text completions.

Providers are specified via strings of the form `<provider>:<model_name>`, for
example `ollama:llama3`, `openai:gpt-3.5-turbo`, or `anthropic:claude-3-haiku`.
"""
from __future__ import annotations

import os
from typing import Tuple
from .prompt import SYSTEM_PROMPT

import httpx


class ModelRouter:
    """Route completion requests to the appropriate model provider.

    The router parses a model specification string and exposes a single
    asynchronous method ``completions`` that takes a prompt and returns the
    generated response.  If no provider is recognised, the router will return
    the prompt itself as a fallback.
    """

    def __init__(self, model_spec: str) -> None:
        # Split the model specification into provider and model name.  If no
        # provider prefix is given assume 'echo', which simply returns the
        # prompt unchanged.
        if ":" in model_spec:
            provider, model = model_spec.split(":", 1)
        else:
            provider, model = "echo", model_spec
        self.provider: str = provider.lower()
        self.model: str = model

    async def completions(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate a completion for a given prompt using the configured provider.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        max_tokens : int, optional
            Maximum number of tokens to return.  Only used for API providers
            that support this parameter.  Defaults to provider defaults.

        Returns
        -------
        str
            The generated text from the model.  If the provider is not known
            the prompt is returned unchanged.
        """
        # Dispatch based on provider
        if self.provider == "openai":
            return await self._call_openai(prompt, max_tokens)
        if self.provider == "anthropic":
            return await self._call_anthropic(prompt, max_tokens)
        if self.provider == "ollama":
            return await self._call_ollama(prompt)
        # Echo provider: simply return the prompt
        return prompt

    async def _call_openai(self, prompt: str, max_tokens: int | None) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "[OPENAI] missing OPENAI_API_KEY"
        url = "https://api.openai.com/v1/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Basic chat payload; adjust model_name in config
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens or 300,
        }
        timeout = httpx.Timeout(30.0, read=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                    url,
                    headers=headers,
                    json=payload,)
            if r.status_code != 200:
                return f"[OPENAI ERROR] {r.status_code}: {r.text}"
            data = r.json()
            return data["choices"][0]["message"]["content"]

    async def _call_anthropic(self, prompt: str, max_tokens: int | None) -> str:
        """Call the Anthropic messages endpoint."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "[ANTHROPIC_API_KEY environment variable not set]"
        # Anthropic uses a different payload structure; messages is a list of
        # dictionaries with ``role`` and ``content`` keys.  The model
        # specification includes the version after the colon.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or 800,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            # The response content is an array of messages; the first element
            # contains the assistant's text in the "text" field.
            return data["content"][0]["text"]

    async def _call_ollama(self, prompt: str) -> str:
        """Call a locally running Ollama model."""
        # Ollama must be installed and running on the local machine (default
        # port 11434).  The /api/generate endpoint supports basic prompt and
        # model parameters.
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                )
                data = response.json()
                return data.get("response", "")
            except Exception:
                return "[Failed to call Ollama; is the service running?]"