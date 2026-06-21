#!/usr/bin/env python3
"""
Labnana Image Generation Backend

Generates images via the Labnana gateway (https://labnana.com/docs/openapi/guide).

Labnana is NOT OpenAI-compatible despite the "openapi" path segment:
  - Endpoint:  POST {base}/images/generation   (singular, not /images/generations)
  - Request:   private schema — requires a `provider` field plus an `imageConfig`
               object, instead of OpenAI's flat `size`/`n` parameters
  - Response:  Google Gemini shape — candidates[].content.parts[].inlineData.data

Configuration keys (process env wins, `.env` is the fallback layer):
  LABNANA_API_KEY   (required) API key, e.g. lh_sk_...
  LABNANA_MODEL     (optional) Model name (default: gpt-image-2)
  LABNANA_BASE_URL  (optional) Gateway base (default: https://api.labnana.com/openapi/v1)
  LABNANA_PROVIDER  (optional) Force provider: openai or google.
                    When unset it is derived from the model name.

For convenience, OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL are accepted as
fallbacks for the matching LABNANA_* keys, so an existing OpenAI-style .env that
points at Labnana keeps working.

Allowed models (per Labnana docs):
  gpt-image-2            -> provider openai
  gemini-3-pro-image     -> provider google
  gemini-3.1-flash-image -> provider google

Dependencies:
  pip install requests Pillow
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    print('Use via: python3 skills/ppt-master/scripts/image_gen.py "prompt" --backend labnana')
    raise SystemExit(0 if any(arg in {"-h", "--help", "help"} for arg in sys.argv[1:]) else 1)

import base64
import os
import time

import requests
from image_backends.backend_common import (
    MAX_RETRIES,
    http_error,
    is_rate_limit_error,
    normalize_image_size,
    resolve_output_path,
    retry_delay,
    save_image_bytes,
)


DEFAULT_MODEL = "gpt-image-2"
DEFAULT_BASE_URL = "https://api.labnana.com/openapi/v1"

# Labnana imageConfig.imageSize accepts only these; the skill's 512px maps to 1K.
SUPPORTED_IMAGE_SIZES = {"1K", "2K", "4K"}

# Aspect ratios accepted by all Labnana models.
SUPPORTED_ASPECT_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"}
# Extra ratios only the flash model accepts.
FLASH_ONLY_ASPECT_RATIOS = {"1:4", "4:1", "1:8", "8:1"}
# Ratios the unified CLI allows but Labnana does not — mapped to the nearest one.
ASPECT_RATIO_FALLBACKS = {"4:5": "3:4", "5:4": "4:3"}

VALID_PROVIDERS = {"openai", "google"}

MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}


def _resolve_provider(model: str) -> str:
    """Pick the Labnana `provider` from env override or the model name."""
    override = (os.environ.get("LABNANA_PROVIDER") or "").strip().lower()
    if override:
        if override not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid LABNANA_PROVIDER='{override}'. "
                f"Supported: {', '.join(sorted(VALID_PROVIDERS))}."
            )
        return override
    normalized = (model or "").strip().lower()
    if normalized.startswith("gemini"):
        return "google"
    return "openai"


def _resolve_aspect_ratio(model: str, aspect_ratio: str) -> str:
    """Map the requested aspect ratio onto a Labnana-supported value."""
    if aspect_ratio in SUPPORTED_ASPECT_RATIOS:
        return aspect_ratio
    is_flash = (model or "").strip().lower().startswith("gemini-3.1-flash")
    if is_flash and aspect_ratio in FLASH_ONLY_ASPECT_RATIOS:
        return aspect_ratio
    if aspect_ratio in ASPECT_RATIO_FALLBACKS:
        return ASPECT_RATIO_FALLBACKS[aspect_ratio]
    raise ValueError(
        f"Unsupported aspect ratio '{aspect_ratio}' for Labnana backend. "
        f"Supported: {sorted(SUPPORTED_ASPECT_RATIOS)}"
    )


def _resolve_image_size(image_size: str) -> str:
    """Map the requested image size onto a Labnana-supported value (512px -> 1K)."""
    normalized = normalize_image_size(image_size)
    if normalized in SUPPORTED_IMAGE_SIZES:
        return normalized
    return "1K"


def _generation_url(base_url: str | None) -> str:
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if base.endswith("/images/generation"):
        return base
    return f"{base}/images/generation"


def _extract_inline_image(resp: dict) -> tuple[bytes, str]:
    """Pull the first inline image (bytes, ext) out of a Gemini-shaped response."""
    candidates = resp.get("candidates") if isinstance(resp, dict) else None
    if not candidates:
        raise RuntimeError(
            "No image was generated. The gateway returned no candidates "
            f"(response keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__})."
        )
    for candidate in candidates:
        parts = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = (inline.get("mimeType") or inline.get("mime_type") or "image/png").lower()
                ext = MIME_TO_EXT.get(mime, ".png")
                return base64.b64decode(inline["data"]), ext
    raise RuntimeError(
        "No inline image found in the gateway response. The model may have "
        "refused the request or returned text only."
    )


def _post_generation(api_key: str, base_url: str | None, request: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        _generation_url(base_url),
        headers=headers,
        json=request,
        timeout=300,
    )
    if not response.ok:
        raise http_error(response, "Labnana image generation")
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Labnana image generation returned invalid JSON.") from exc


def _generate_image(api_key: str, prompt: str, aspect_ratio: str, image_size: str,
                    output_dir: str | None, filename: str | None,
                    model: str, base_url: str | None) -> str:
    provider = _resolve_provider(model)
    ratio = _resolve_aspect_ratio(model, aspect_ratio)
    size = _resolve_image_size(image_size)

    request = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "imageConfig": {
            "imageSize": size,
            "aspectRatio": ratio,
        },
    }

    mode_label = f"Proxy: {base_url}" if base_url else "Labnana"
    print(f"[Labnana - {mode_label}]")
    print(f"  Provider:     {provider}")
    print(f"  Model:        {model}")
    print(f"  Prompt:       {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"  Aspect ratio: {ratio} (requested {aspect_ratio})")
    print(f"  Image size:   {size}")
    print()

    start_time = time.time()
    print("  [..] Generating...", flush=True)
    resp = _post_generation(api_key, base_url, request)
    elapsed = time.time() - start_time
    print(f"  [DONE] Image generated ({elapsed:.1f}s)")

    image_bytes, ext = _extract_inline_image(resp)
    path = resolve_output_path(prompt, output_dir, filename, ext)
    return save_image_bytes(image_bytes, path)


def generate(prompt: str,
             aspect_ratio: str = "1:1", image_size: str = "1K",
             output_dir: str = None, filename: str = None,
             model: str = None, max_retries: int = MAX_RETRIES) -> str:
    """
    Labnana image generation with automatic retry.

    Reads credentials from the current process environment or a `.env` file:
      LABNANA_API_KEY  (or OPENAI_API_KEY as fallback)
      LABNANA_BASE_URL (or OPENAI_BASE_URL as fallback)
      LABNANA_MODEL    (or OPENAI_MODEL as fallback)

    Returns:
        Path of the saved image file.
    """
    api_key = os.environ.get("LABNANA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LABNANA_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "No API key found. Set LABNANA_API_KEY (or OPENAI_API_KEY) in the "
            "current environment or a .env file."
        )

    if model is None:
        model = (
            os.environ.get("LABNANA_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_MODEL
        )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return _generate_image(api_key, prompt, aspect_ratio, image_size,
                                   output_dir, filename, model, base_url)
        except (ValueError, FileNotFoundError):
            # Configuration / mapping errors are not worth retrying.
            raise
        except Exception as e:  # noqa: BLE001 — network/server errors are retryable
            last_error = e
            if attempt < max_retries and is_rate_limit_error(e):
                delay = retry_delay(attempt, rate_limited=True)
                print(f"\n  [WARN] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                      f"Waiting {delay}s before retry...")
                time.sleep(delay)
            elif attempt < max_retries:
                delay = retry_delay(attempt, rate_limited=False)
                print(f"\n  [WARN] Error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                      f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                break

    raise RuntimeError(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")
