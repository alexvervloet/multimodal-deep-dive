"""
multimodal/providers.py: the ONLY provider-specific file.

Multimodality is mostly an *architecture* question, which modality goes in which
slot, but unlike the other repos in this series, the providers genuinely differ
in *capability*, not just in request shape. That's the honest, load-bearing fact
of this repo:

  capability          openai                         claude
  ----------          ------                         ------
  vision (image in)   YES (gpt-6-luna)                YES (claude-haiku-4-5)
  audio in  (STT)     YES (gpt-transcribe)           NO  (no native audio API)
  audio out (TTS)     YES (gpt-4o-mini-tts)          NO
  image generation    YES (gpt-image-2.5-flare)     NO

So this file does two things:

  1. Normalizes the parts that BOTH providers can do (vision chat) to one
     interface, `chat()`, that takes a list of content blocks. The only thing
     that differs is the shape of an image block, which `image_block()` builds.
  2. Exposes single-provider features (`transcribe`, `speak`, `generate_image`)
     that raise a clear, friendly error when the active provider can't do them,
     so the examples degrade gracefully instead of crashing.

Model IDs mirror the sibling repos (see ../rag-deep-dive/rag/providers.py, ../agents-deep-dive/agent/providers.py):
OpenAI `gpt-6-luna`, Claude `claude-haiku-4-5`: the cheap, fast workhorses. Luna
reasons by default; `chat()` sends `reasoning_effort="none"` so it answers directly.

Clients are created lazily and cached, so importing this module never forces an
SDK import or a network call.
"""

import base64
import io
import os
import sys
from functools import lru_cache

# --- Models per stack. Mirrors the sibling repos' cheap defaults. -----------
_OPENAI_CHAT = "gpt-6-luna"  # vision-capable chat
_OPENAI_STT = "gpt-transcribe"  # speech-to-text (whisper-1 shuts down 2027-02-26)
# Text-to-speech. Deprecated 2026-10-01, shuts down 2027-01-06, along with tts-1
# and tts-1-hd. OpenAI's named replacement, gpt-realtime-2.1-mini, only runs over
# the Realtime API (WebSocket/WebRTC), not the audio.speech endpoint used here, and
# as of 2026-10-06 no speech-endpoint successor exists. Revisit before January.
_OPENAI_TTS = "gpt-4o-mini-tts"
_OPENAI_IMAGE = "gpt-image-2.5-flare"  # image generation (gpt-image-1 shut down 2026-10-23)
_CLAUDE_CHAT = "claude-haiku-4-5"  # vision-capable chat

_KEYS = {"openai": ["OPENAI_API_KEY"], "claude": ["ANTHROPIC_API_KEY"]}

# Which optional capabilities each provider has. The examples consult this to
# decide whether to run for real or print an honest "not supported here" note.
_CAPABILITIES = {
    "openai": {"vision": True, "stt": True, "tts": True, "image_gen": True, "pdf": True},
    "claude": {"vision": True, "stt": False, "tts": False, "image_gen": False, "pdf": True},
}


class UnsupportedCapability(RuntimeError):
    """Raised when the active provider can't do the requested modality.

    We raise a typed error (not a bare RuntimeError) so the examples can catch it
    and print a friendly message instead of a traceback. This is how the repo
    stays honest: a Claude user running the audio example gets a clear
    explanation, not a crash."""


# ---------------------------------------------------------------------------
# Provider identity & readiness: same pattern as every sibling repo.
# ---------------------------------------------------------------------------
def provider_name() -> str:
    """The active stack: 'openai' (default) or 'claude'. Set via PROVIDER in .env."""
    return os.getenv("PROVIDER", "openai").strip().lower()


def required_keys() -> list[str]:
    return _KEYS.get(provider_name(), [])


def supports(capability: str) -> bool:
    """True if the active provider can do `capability`
    ('vision' | 'stt' | 'tts' | 'image_gen')."""
    return _CAPABILITIES.get(provider_name(), {}).get(capability, False)


def describe() -> str:
    """One-line summary of the active stack, handy for examples to print."""
    p = provider_name()
    if p == "openai":
        return f"openai  (chat={_OPENAI_CHAT}, stt={_OPENAI_STT}, tts={_OPENAI_TTS}, image={_OPENAI_IMAGE})"
    if p == "claude":
        return f"claude  (chat={_CLAUDE_CHAT}; vision yes, audio/image-gen no)"
    return f"unknown provider {p!r}"


def ensure_ready() -> None:
    """Fail fast with a friendly message if the stack isn't configured.

    Call this at the top of any script that makes a real API call, *after*
    `load_dotenv()`."""
    p = provider_name()
    if p not in _KEYS:
        sys.exit(f"PROVIDER={p!r} is not recognized. Set PROVIDER=openai or claude in .env.")
    missing = [k for k in required_keys() if not os.getenv(k)]
    if missing:
        sys.exit(
            f"PROVIDER={p} needs {', '.join(missing)} in the environment. "
            f"Provide them via secrun (see ../docs/SECRETS.md), or run `secrun python check_setup.py`."
        )


# ---------------------------------------------------------------------------
# Lazy, cached clients.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Content blocks: the heart of "put the right modality in the right slot."
#
# A multimodal message is not a string; it's a LIST of typed blocks. A text
# block and an image block sit side by side in one user turn. The two providers
# spell an image block differently, so we build it here and the examples never
# have to care which provider is active.
# ---------------------------------------------------------------------------
def text_block(text: str) -> dict:
    """A text content block, identical shape on both providers."""
    return {"type": "text", "text": text}


def image_block(data: bytes, media_type: str = "image/png") -> dict:
    """An image content block from raw image bytes, in the active provider's shape.

    Both providers take base64-encoded image bytes; they just wrap them
    differently. OpenAI uses an `image_url` with a `data:` URI; Claude uses an
    `image` block with a typed `source`. Same picture, two envelopes."""
    b64 = base64.standard_b64encode(data).decode("ascii")
    p = provider_name()
    if p == "openai":
        # Set detail on purpose. On gpt-6-luna the default ("auto") doesn't shrink
        # large images, so a 4K screenshot costs over three times what "high"
        # does. See multimodal/tokens.py for the measured numbers.
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "high"},
        }
    if p == "claude":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }
    raise ValueError(f"Unknown PROVIDER={p!r}.")


def pdf_block(data: bytes, filename: str = "document.pdf") -> dict:
    """A NATIVE PDF content block from raw PDF bytes, in the active provider's shape.

    This is the key contrast with §4. There, we turned a document into a picture
    (a screenshot) and used vision: the workaround. A native PDF block hands the
    model the *document itself*: it reads the real text, keeps page structure, and
    handles many pages at once. It's the input enterprise document pipelines
    actually use.

    Both providers take base64-encoded PDF bytes and, again, only the envelope
    differs. OpenAI passes a `file` part with a `data:` URI (like an image URL);
    Claude uses a `document` block with a typed `source`. The PDF rides in the
    *same user turn* as your question; a document is just another slot."""
    b64 = base64.standard_b64encode(data).decode("ascii")
    p = provider_name()
    if p == "openai":
        return {
            "type": "file",
            "file": {"filename": filename, "file_data": f"data:application/pdf;base64,{b64}"},
        }
    if p == "claude":
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
        }
    raise ValueError(f"Unknown PROVIDER={p!r}.")


def chat(system: str, content_blocks: list[dict], max_tokens: int = 1024) -> str:
    """Send a system prompt + a list of user content blocks; return the text reply.

    `content_blocks` is a mix of `text_block(...)` and `image_block(...)`. This is
    the one call both providers share. The shape difference is only in *where* the
    system prompt goes (OpenAI: a message; Claude: a top-level `system=`) and how
    the reply is unpacked, both normalized to a plain string here.

    Requires the `vision` capability (both providers have it)."""
    p = provider_name()
    if p == "openai":
        resp = _openai_client().chat.completions.create(
            model=_OPENAI_CHAT,
            reasoning_effort="none",  # luna thinks by default; this is a describe task
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content_blocks},  # type: ignore[arg-type]
            ],
        )
        return resp.choices[0].message.content or ""
    if p == "claude":
        resp = _anthropic_client().messages.create(
            model=_CLAUDE_CHAT,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content_blocks}],  # type: ignore[arg-type]
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    raise ValueError(f"Unknown PROVIDER={p!r}.")


# ---------------------------------------------------------------------------
# Single-provider capabilities. Each one checks `supports(...)` first and raises
# a clear UnsupportedCapability if the active provider can't do it. The examples
# catch that and print an honest note: no crashes, no pretending.
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Speech-to-text. OpenAI only. Claude has no native audio API."""
    if not supports("stt"):
        raise UnsupportedCapability(
            f"PROVIDER={provider_name()} has no speech-to-text API. "
            f"Use PROVIDER=openai for transcription."
        )
    f = io.BytesIO(audio_bytes)
    f.name = filename  # the SDK infers the format from the extension
    resp = _openai_client().audio.transcriptions.create(model=_OPENAI_STT, file=f)
    return resp.text


def speak(text: str, voice: str = "alloy") -> bytes:
    """Text-to-speech -> MP3 bytes. OpenAI only. Claude has no TTS API."""
    if not supports("tts"):
        raise UnsupportedCapability(
            f"PROVIDER={provider_name()} has no text-to-speech API. "
            f"Use PROVIDER=openai for speech synthesis."
        )
    resp = _openai_client().audio.speech.create(
        model=_OPENAI_TTS, voice=voice, input=text
    )
    return resp.content  # raw audio bytes (MP3)


def generate_image(prompt: str, size: str = "1024x1024", quality: str = "low") -> bytes:
    """Text-to-image -> PNG bytes. OpenAI only (gpt-image-2.5-flare). Claude can't
    generate images; it's vision-in only.

    `quality` is set on purpose. The API's default, "auto", lets the model pick,
    and the levels run up to "max". Measured 2026-10-05 on a 1024x1024 image: "low"
    cost about $0.006 (196 output tokens) and "medium" about $0.013 (439 tokens)."""
    if not supports("image_gen"):
        raise UnsupportedCapability(
            f"PROVIDER={provider_name()} cannot generate images (it's vision-in only). "
            f"Use PROVIDER=openai for image generation (gpt-image-2.5-flare)."
        )
    resp = _openai_client().images.generate(
        model=_OPENAI_IMAGE, prompt=prompt, size=size, quality=quality  # type: ignore[arg-type]
    )
    assert resp.data is not None
    return base64.b64decode(resp.data[0].b64_json or "")
