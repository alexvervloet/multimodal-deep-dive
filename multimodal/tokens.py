"""
multimodal/tokens.py: how an image becomes tokens (offline, no key).

The single most surprising thing about multimodal models: an image is not free,
and it is not one token. It is *tokenized* into a number of tokens that depends on
its pixel dimensions, and a big screenshot can cost more than a paragraph of
text. This module estimates that cost with pure arithmetic, so you can reason
about (and budget for) image inputs WITHOUT making an API call.

These are documented, public formulas, but they're approximations and providers
change them, so treat this as a teaching tool, not a billing source of truth. The
real number always comes back in the API response's usage field.

Two provider models, two different schemes:

  OpenAI (gpt-4o family): a base cost + a per-tile cost. The image is scaled to
  fit a budget, then chopped into 512x512 tiles; each tile costs a fixed number
  of tokens, plus one base amount.

  Claude: tokens ≈ (width * height) / 750, capped: a simple area-based rule.

The exact constants below match each provider's published guidance at the time of
writing; the point is the *shape* of the cost, which is stable: tokens scale with
pixels, and a downscale is the cheapest optimization you have.
"""

import math
from dataclasses import dataclass

# --- OpenAI gpt-4o-mini "high detail" tiling constants --------------------
_OPENAI_BASE_TOKENS = 2833  # base tokens for gpt-4o-mini (higher than gpt-4o)
_OPENAI_TILE_TOKENS = 5667  # per 512x512 tile, gpt-4o-mini
_OPENAI_MAX_SIDE = 2048  # image is first shrunk to fit a 2048x2048 box
_OPENAI_SHORT_SIDE = 768  # then the shortest side is shrunk to 768
_OPENAI_TILE = 512

# --- Claude area rule -----------------------------------------------------
_CLAUDE_TOKENS_PER_PIXEL = 1 / 750  # tokens ≈ (w*h)/750
_CLAUDE_MAX_TOKENS = 1600  # very large images are capped/recommended-resized


@dataclass
class ImageCost:
    """The estimated token cost of one image, plus how it was derived."""

    provider: str
    width: int
    height: int
    tokens: int
    explanation: str


def openai_image_tokens(width: int, height: int) -> ImageCost:
    """Estimate gpt-4o-mini image tokens for a width x height image (high detail).

    The algorithm: shrink to fit a 2048 box, then shrink so the short side is 768,
    then count 512x512 tiles. Cost = base + tiles * per-tile."""
    w, h = width, height
    # 1. Fit within a 2048x2048 box.
    if max(w, h) > _OPENAI_MAX_SIDE:
        scale = _OPENAI_MAX_SIDE / max(w, h)
        w, h = round(w * scale), round(h * scale)
    # 2. Scale the shortest side down to 768.
    if min(w, h) > _OPENAI_SHORT_SIDE:
        scale = _OPENAI_SHORT_SIDE / min(w, h)
        w, h = round(w * scale), round(h * scale)
    # 3. Count 512x512 tiles (round up).
    tiles = math.ceil(w / _OPENAI_TILE) * math.ceil(h / _OPENAI_TILE)
    tokens = _OPENAI_BASE_TOKENS + tiles * _OPENAI_TILE_TOKENS
    return ImageCost(
        provider="openai",
        width=width,
        height=height,
        tokens=tokens,
        explanation=(
            f"scaled to {w}x{h}, {tiles} tile(s) of {_OPENAI_TILE}px -> "
            f"{_OPENAI_BASE_TOKENS} base + {tiles}*{_OPENAI_TILE_TOKENS}"
        ),
    )


def claude_image_tokens(width: int, height: int) -> ImageCost:
    """Estimate Claude image tokens: ~(width * height) / 750, with a soft cap."""
    raw = (width * height) * _CLAUDE_TOKENS_PER_PIXEL
    tokens = min(int(raw), _CLAUDE_MAX_TOKENS)
    capped = " (capped; resize recommended)" if raw > _CLAUDE_MAX_TOKENS else ""
    return ImageCost(
        provider="claude",
        width=width,
        height=height,
        tokens=tokens,
        explanation=f"({width}*{height})/750 = {raw:.0f} tokens{capped}",
    )


def estimate(provider: str, width: int, height: int) -> ImageCost:
    """Estimate image tokens for the named provider ('openai' | 'claude')."""
    if provider == "openai":
        return openai_image_tokens(width, height)
    if provider == "claude":
        return claude_image_tokens(width, height)
    raise ValueError(f"Unknown provider {provider!r} (expected 'openai' or 'claude').")
