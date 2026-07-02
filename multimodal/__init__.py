"""
multimodal — a from-scratch toolkit for putting images and audio into an LLM.
=============================================================================

The whole repo hangs off one idea: a multimodal model takes more than text in its
context — images and audio — and the skill is putting the right modality in the
right slot, then paying attention to what each one costs.

This package is small on purpose. Read it top to bottom:

  providers.py  — the ONE provider-specific file: vision chat (both providers),
                  plus single-provider audio/image-gen with graceful errors.
  media.py      — dependency-free helpers to load/save images and audio.
  tokens.py     — estimate an image's token cost offline (the §9 "token math").

The examples/ and hands_on/ scripts import from here; the README walks them in
order, each ending in something you can run.
"""

from . import media, providers, tokens
from .providers import (
    UnsupportedCapability,
    chat,
    describe,
    ensure_ready,
    generate_image,
    image_block,
    pdf_block,
    provider_name,
    speak,
    supports,
    text_block,
    transcribe,
)

__all__ = [
    "media",
    "providers",
    "tokens",
    "UnsupportedCapability",
    "chat",
    "describe",
    "ensure_ready",
    "generate_image",
    "image_block",
    "pdf_block",
    "provider_name",
    "speak",
    "supports",
    "text_block",
    "transcribe",
]
