from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self, messages: list[dict], stream: bool = False, **kwargs
    ) -> str | AsyncIterator[str]:
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        import openai

        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    async def chat(self, messages, stream=False, **kwargs):
        if stream:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **kwargs
            )

            async def _stream():
                async for chunk in resp:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content

            return _stream()

        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content or ""

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.embeddings.create(
            model=self.embed_model, input=text
        )
        return resp.data[0].embedding


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        import anthropic

        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    async def chat(self, messages, stream=False, **kwargs):
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                filtered.append({"role": m["role"], "content": m["content"]})

        if stream:
            resp = await self.client.messages.create(
                model=self.model,
                system=system.strip() or None,
                messages=filtered,
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=True,
            )

            async def _stream():
                async for chunk in resp:
                    if chunk.type == "content_block_delta" and chunk.delta.text:
                        yield chunk.delta.text

            return _stream()

        resp = await self.client.messages.create(
            model=self.model,
            system=system.strip() or None,
            messages=filtered,
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return resp.content[0].text if resp.content else ""

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Anthropic does not provide embeddings via API")


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        import httpx

        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self._httpx = httpx.AsyncClient(timeout=120)

    async def chat(self, messages, stream=False, **kwargs):
        if stream:
            async with self._httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
            ) as resp:
                async def _stream():
                    async for line in resp.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if data.get("done"):
                                break
                            if data.get("message", {}).get("content"):
                                yield data["message"]["content"]

                return _stream()

        resp = await self._httpx.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
        )
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def embed(self, text: str) -> list[float]:
        resp = await self._httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        data = resp.json()
        return data.get("embedding", [])

    async def close(self):
        await self._httpx.aclose()


def create_provider(
    provider: str = "openai", **kwargs
) -> LLMProvider:
    match provider.lower():
        case "openai":
            return OpenAIProvider(**kwargs)
        case "anthropic":
            return AnthropicProvider(**kwargs)
        case "ollama":
            return OllamaProvider(**kwargs)
        case _:
            raise ValueError(f"Provider desconhecido: {provider}")
