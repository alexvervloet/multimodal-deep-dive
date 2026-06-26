"""
Example 05 — audio IN: speech-to-text (transcription).
======================================================

A new modality, a new slot. Audio doesn't go in a chat content block — it goes to
a dedicated transcription endpoint (OpenAI's Whisper). You hand it audio bytes;
you get back text. That text can then flow into any text or vision prompt — which
is exactly what the voice-Q&A capstone does.

  ⚠️  PROVIDER SUPPORT: This is OpenAI-only. Claude has NO native audio API.
      With PROVIDER=claude, this example explains that and exits cleanly — it
      does not crash. That honesty is the whole point of the capability table in
      multimodal/providers.py.

The sample clip (assets/note.wav) is a self-made 440 Hz tone, not real speech —
so a perfect transcription would be empty or near-empty. That's fine: the goal
here is to see the *request shape* and the round-trip, not to marvel at accuracy.
Point it at a real voice recording to see real text.

Run it:

    python examples/05_transcribe_audio.py
    python examples/05_transcribe_audio.py path/to/your/voice.mp3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from multimodal import media, providers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
providers.ensure_ready()
print(f"Provider: {providers.describe()}\n")

# Degrade gracefully: if the active provider can't do speech-to-text, say so.
if not providers.supports("stt"):
    print(
        f"PROVIDER={providers.provider_name()} has no speech-to-text API.\n"
        f"Claude is vision-only — it cannot transcribe audio. To run this example,\n"
        f"set PROVIDER=openai in .env (Whisper). Skipping the call; nothing went wrong."
    )
    sys.exit(0)

audio_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "note.wav")
audio_data, _ = media.load_bytes(audio_path)
filename = os.path.basename(audio_path)

print(f"Audio: {os.path.relpath(audio_path, ROOT)}  ({len(audio_data):,} bytes)")
print("Transcribing with Whisper...\n")

text = providers.transcribe(audio_data, filename=filename)

print("Transcript:")
print(repr(text) if text.strip() else "(empty — the sample is a tone, not speech)")

print(
    "\nThat text is now ordinary text you can feed into any prompt. The capstone "
    "chains transcribe -> ask so you can talk to a model and get a spoken-question "
    "answered. Next (example 06): the reverse direction — text OUT as audio."
)
