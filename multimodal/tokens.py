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

  OpenAI (gpt-6-luna, and gpt-5.4-nano before it): about 1.2 tokens per 32x32
  pixel patch. The `detail` setting decides how many patches you can be billed
  for: "high" shrinks big images to fit 2,500 patches, "low" to 256, and "auto"
  (the default) doesn't shrink at all on luna.

  Claude: tokens ≈ (width * height) / 750, capped: a simple area-based rule.

The OpenAI constants below were measured against the live API on 2026-10-03
(`usage.prompt_tokens` minus a text-only baseline) and are pinned by
tests/test_tokens.py. They replaced tile constants that put a 512x512 image at
8,500 tokens; both models bill 307. The Claude rule follows Anthropic's published
guidance. The *shape* of the cost is what lasts: tokens scale with pixels, and a
downscale is the cheapest optimization you have.
"""

import math
from dataclasses import dataclass

# --- OpenAI patch rule (gpt-6-luna; gpt-5.4-nano measured the same) -------
_OPENAI_PATCH = 32  # the image is cut into 32x32-pixel patches
_OPENAI_TOKENS_PER_PATCH = 1.2
_OPENAI_PATCH_BUDGET = {"high": 2_500, "low": 256}  # "auto" has no budget on luna
_OPENAI_MAX_PATCHES = 30_000  # past this, luna returns a 400 instead of resizing

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


def _patches(width: float, height: float) -> int:
    return math.ceil(width / _OPENAI_PATCH) * math.ceil(height / _OPENAI_PATCH)


def openai_image_tokens(width: int, height: int, detail: str = "high") -> ImageCost:
    """Estimate gpt-6-luna image tokens for a width x height image.

    The algorithm: count 32x32 patches. If that's over the budget for `detail`,
    shrink the image (keeping its shape) until a whole number of patches fits,
    then count again. Cost = patches * 1.2, rounded down.

    `detail="auto"` is what you get if you leave `detail` out, and on luna it
    applies no budget: a 3000x3000 photo is 8,836 patches, 10,603 tokens. The
    same photo at "high" is 3,000. gpt-5.4-nano treated "auto" like "high", so
    the cheaper model can be the pricier one for big images."""
    if detail not in ("high", "low", "auto"):
        raise ValueError(f"detail must be 'high', 'low', or 'auto', not {detail!r}")
    patches = _patches(width, height)
    budget = _OPENAI_PATCH_BUDGET.get(detail)
    w, h = float(width), float(height)
    if budget is not None and patches > budget:
        # Shrink so the area fits the budget, then a little more so both sides
        # land on a whole number of patches.
        shrink = math.sqrt(_OPENAI_PATCH**2 * budget / (width * height))
        across, down = width * shrink / _OPENAI_PATCH, height * shrink / _OPENAI_PATCH
        shrink *= min(math.floor(across) / across, math.floor(down) / down)
        w, h = width * shrink, height * shrink
        # One side now lands exactly on a patch boundary; the other rounds up.
        # The epsilon keeps float noise on the exact side from adding a patch.
        patches = math.ceil(w / _OPENAI_PATCH - 1e-6) * math.ceil(h / _OPENAI_PATCH - 1e-6)
    if patches > _OPENAI_MAX_PATCHES:
        raise ValueError(
            f"{width}x{height} at detail={detail!r} is {patches:,} patches; luna rejects "
            f"anything over {_OPENAI_MAX_PATCHES:,}. Resize it, or use detail='high'."
        )
    tokens = int(patches * _OPENAI_TOKENS_PER_PATCH)
    scaled = f"scaled to {round(w)}x{round(h)}, " if (round(w), round(h)) != (width, height) else ""
    return ImageCost(
        provider="openai",
        width=width,
        height=height,
        tokens=tokens,
        explanation=f"detail={detail}: {scaled}{patches:,} patches x {_OPENAI_TOKENS_PER_PATCH}",
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
