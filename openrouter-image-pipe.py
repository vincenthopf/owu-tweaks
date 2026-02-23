"""
title: OpenRouter: Image Pipe
description: Generate and edit images via OpenRouter with automatic model selection
author: Vincent Hopf
author_url: https://github.com/vincenthopf
funding_url: https://github.com/vincenthopf
version: 1.1
created: 2025-12-18
"""

import aiohttp
import base64
import hashlib
import json
import os
import pydantic
import time
import typing
from pathlib import Path


# Image icon (Heroicons outline)
ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHJlY3QgeD0iMyIgeT0iMyIgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4IiByeD0iMiIgcnk9IjIiLz48Y2lyY2xlIGN4PSI5IiBjeT0iOSIgcj0iMiIvPjxwYXRoIGQ9Im0yMSAxNS0zLjA4Ni0zLjA4NmEyIDIgMCAwIDAtMi44MjggMEw2IDIxIi8+PC9zdmc+"

CACHE_DIR = Path("/app/backend/data/cache/image_generations")
CACHE_FILE = CACHE_DIR / "models_cache.json"
DEFAULT_MODEL = "google/gemini-2.5-flash-image"

# Well-defined quality variants using only factual API data
QUALITY_VARIANTS = {
    "cheapest": {
        "sort_key": lambda m: float(m["pricing"]["prompt"])
        + float(m["pricing"]["completion"]),
        "sort_reverse": False,
    },
    "largest-context": {
        "sort_key": lambda m: m["context_length"],
        "sort_reverse": True,
    },
    "premium": {
        "sort_key": lambda m: float(m["pricing"]["prompt"])
        + float(m["pricing"]["completion"]),
        "sort_reverse": True,
    },
}

DEFAULT_PRESETS = """google-cheapest|Google Cheapest|google|cheapest
google-premium|Google Premium|google|premium
openai-cheapest|OpenAI Cheapest|openai|cheapest
openai-premium|OpenAI Premium|openai|premium"""


class Pipe:
    """
    Image generation pipe for OpenRouter. Saves images to disk and returns URLs
    to avoid browser SSE "Chunk too big" errors with base64 data.

    Features:
    - Multiple preset variants (Google/OpenAI × Cheapest/Premium)
    - Zero-maintenance model selection via OpenRouter API
    - Custom model override for advanced users
    - Aspect ratio control (Gemini models)
    - Automatic image editing detection (when user uploads images)
    - Images served via Open WebUI's /cache/ endpoint
    """

    class Valves(pydantic.BaseModel):
        provider: typing.Literal["any", "google", "openai"] = pydantic.Field(
            default="any",
            description="Provider for base model (preset variants ignore this)",
        )
        quality: typing.Literal["cheapest", "largest-context", "premium"] = (
            pydantic.Field(
                default="cheapest",
                description="Quality for base model (preset variants ignore this)",
            )
        )
        custom_model: str = pydantic.Field(
            default="",
            description="Override base model with specific ID (preset variants ignore this)",
        )
        aspect_ratio: typing.Literal[
            "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9", "5:4", "4:5"
        ] = pydantic.Field(
            default="1:1",
            description="Image aspect ratio (applies to all variants, Gemini only)",
        )
        custom_aspect_ratio: str = pydantic.Field(
            default="",
            description="Custom aspect ratio override (e.g., '7:3')",
        )
        cache_hours: float = pydantic.Field(
            default=12.0,
            ge=0.0,
            le=168.0,
            description="Hours to cache model list (0 = no cache)",
        )
        presets: str = pydantic.Field(
            default=DEFAULT_PRESETS,
            description="""Model presets (one per line, pipe-separated):
id|Display Name|provider|quality|model (optional)

Example:
fast|Fast Generation|google|cheapest
best|Best Quality|openai|premium
gemini3|Gemini 3 Pro|google|premium|google/gemini-3-pro-image-preview""",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        """Return available model variants."""
        variants = [
            {"id": "openrouter-image", "name": "OpenRouter: Image Pipe (Default)"}
        ]

        for line in self.valves.presets.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                variants.append(
                    {
                        "id": f"openrouter-image-{parts[0]}",
                        "name": f"OpenRouter: Image Pipe ({parts[1]})",
                    }
                )

        return variants

    def _resolve_preset(self, model_id: str) -> tuple[str, str, str]:
        """Get (provider, quality, custom_model) for selected model variant."""
        # Open WebUI prefixes model IDs with "pipename." - extract the suffix
        if "." in model_id:
            model_id = model_id.split(".", 1)[1]

        if model_id == "openrouter-image":
            return self.valves.provider, self.valves.quality, self.valves.custom_model

        for line in self.valves.presets.strip().split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and model_id == f"openrouter-image-{parts[0]}":
                custom_model = parts[4] if len(parts) >= 5 else ""
                return parts[2], parts[3], custom_model

        return "any", "cheapest", ""

    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __request__=None,
    ) -> typing.AsyncGenerator[str, None]:
        """
        Generate or edit images via OpenRouter API.
        Saves images to disk and yields markdown image references.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        api_key = self._get_api_key(__request__)
        if not api_key:
            yield "Error: No API key configured. Set OPENAI_API_KEY or configure OpenRouter in Open WebUI."
            return

        selected_model_id = body.get("model", "")
        provider, quality, custom_model = self._resolve_preset(selected_model_id)

        # Allow filter override via body fields
        provider = body.pop("_provider", None) or provider
        quality = body.pop("_quality", None) or quality

        if custom_model.strip():
            model = custom_model.strip()
        else:
            model = await self._resolve_model_by_preset(provider, quality)

        aspect_ratio = self._get_aspect_ratio()

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating with {model}...",
                        "done": False,
                    },
                }
            )

        modalities = await self._get_output_modalities(model)

        payload = {
            "model": model,
            "messages": body.get("messages", []),
            "modalities": modalities,
        }

        if "gemini" in model.lower() and aspect_ratio != "1:1":
            payload["image_config"] = {"aspect_ratio": aspect_ratio}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get("error", {}).get(
                                "message", error_text
                            )
                        except json.JSONDecodeError:
                            error_msg = error_text
                        yield f"Error: {error_msg}"
                        return

                    data = await resp.json()

        except aiohttp.ClientError as e:
            yield f"Error: Connection failed - {e}"
            return

        choices = data.get("choices", [])
        if not choices:
            yield "Error: No response from model"
            return

        message = choices[0].get("message", {})
        text_content = message.get("content", "")
        images = message.get("images", [])

        if text_content:
            yield text_content
            if images:
                yield "\n\n"

        for i, img_obj in enumerate(images, 1):
            image_url = img_obj.get("image_url", {}).get("url", "")
            if not image_url.startswith("data:"):
                continue

            saved_path = self._save_image(image_url)
            if saved_path:
                yield f"![Generated image {i}]({saved_path})\n"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Generation complete", "done": True},
                }
            )

    async def _get_output_modalities(self, model_id: str) -> list[str]:
        """Determine the correct output modalities for a model.

        LLMs with image output (Gemini, GPT) support both ["image", "text"].
        Dedicated image-gen models (FLUX, Seedream, Riverflow) only support ["image"].
        Sending unsupported modalities causes OpenRouter to reject the request.
        """
        models = await self._get_cached_models()
        for m in models:
            if m["id"] == model_id:
                return m.get("output_modalities", ["image"])
        return ["image", "text"]

    async def _resolve_model_by_preset(self, provider: str, quality: str) -> str:
        """Fetch and filter models from OpenRouter API based on provider and quality."""
        models = await self._get_cached_models()

        if provider != "any":
            models = [m for m in models if m["provider"] == provider]

        if not models:
            return DEFAULT_MODEL

        variant = QUALITY_VARIANTS.get(quality, QUALITY_VARIANTS["cheapest"])
        models.sort(key=variant["sort_key"], reverse=variant["sort_reverse"])

        return models[0]["id"]

    async def _get_cached_models(self) -> list[dict]:
        """Fetch image-capable models from OpenRouter, with disk caching."""
        if self.valves.cache_hours > 0 and CACHE_FILE.exists():
            try:
                cache = json.loads(CACHE_FILE.read_text())
                age_hours = (time.time() - cache.get("timestamp", 0)) / 3600
                if age_hours < self.valves.cache_hours:
                    return cache.get("models", [])
            except (json.JSONDecodeError, KeyError):
                pass

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://openrouter.ai/api/v1/models") as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except aiohttp.ClientError:
            return []

        models = [
            {
                "id": m["id"],
                "provider": m["id"].split("/")[0],
                "pricing": m.get("pricing", {}),
                "context_length": m.get("context_length", 0),
                "output_modalities": m.get("architecture", {}).get(
                    "output_modalities", []
                ),
            }
            for m in data.get("data", [])
            if "image" in m.get("architecture", {}).get("output_modalities", [])
        ]

        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({"timestamp": time.time(), "models": models}))

        return models

    def _get_aspect_ratio(self) -> str:
        """Resolve aspect ratio: custom > preset."""
        if self.valves.custom_aspect_ratio.strip():
            return self.valves.custom_aspect_ratio.strip()
        return self.valves.aspect_ratio

    def _get_api_key(self, request) -> str:
        """Get OpenRouter API key from Open WebUI config or environment."""
        if request:
            try:
                api_keys = request.app.state.config.OPENAI_API_KEYS
                base_urls = request.app.state.config.OPENAI_API_BASE_URLS
                for idx, url in enumerate(base_urls):
                    if "openrouter" in url.lower():
                        return api_keys[idx]
                if api_keys:
                    return api_keys[0]
            except (AttributeError, IndexError):
                pass
        return os.getenv("OPENAI_API_KEY", "")

    def _save_image(self, data_url: str) -> str | None:
        """
        Save base64 image to cache directory.
        Returns URL path for Open WebUI's /cache/ endpoint.
        """
        try:
            header, b64_data = data_url.split(",", 1)
            mime_type = header.split(";")[0].split(":")[1]

            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
            }
            ext = ext_map.get(mime_type, ".png")

            image_bytes = base64.b64decode(b64_data)
            file_hash = hashlib.md5(image_bytes).hexdigest()
            filename = f"{file_hash}{ext}"

            file_path = CACHE_DIR / filename
            if not file_path.exists():
                file_path.write_bytes(image_bytes)

            return f"/cache/image_generations/{filename}"

        except (ValueError, base64.binascii.Error):
            return None
