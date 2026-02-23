# owu-tweaks

Custom pipes and configs for Open WebUI.

## OpenRouter Image Pipe

A pipe for generating images via OpenRouter with automatic model selection. Supports both LLM-based image models (Gemini, GPT) and dedicated image-gen models (FLUX, Seedream, Riverflow).

Key difference from the upstream pipe: automatically detects each model's supported output modalities from the OpenRouter API, so dedicated image-gen models that only support `["image"]` output don't fail with `"No endpoints found that support the requested output modalities: image, text"`.

### Features

- Preset-based model selection via `image-models.cfg`
- Dynamic modality detection per model
- Aspect ratio control (Gemini models)
- Disk-cached model list from OpenRouter API
- Images saved to disk and served via Open WebUI's `/cache/` endpoint

### Setup

1. Add `openrouter-image-pipe.py` as a pipe in Open WebUI
2. Configure your OpenRouter API key
3. Optionally paste the contents of `image-models.cfg` into the pipe's presets valve

## image-models.cfg

All available OpenRouter image models with verified model IDs and pricing tiers (`cheapest`, `cheap`, `standard`, `premium`).
