"""
Example 06 — audio OUT: text-to-speech.
=======================================

The mirror image of example 05. Instead of audio in -> text out, this is text in
-> audio out. You hand the TTS endpoint a string and a voice; you get back audio
bytes you can save and play.

  ⚠️  PROVIDER SUPPORT: OpenAI-only, like transcription. Claude has no TTS API.
      With PROVIDER=claude, this example explains that and exits cleanly.

We write the result to out/spoken.mp3 (git-ignored). There's no playback inside
this script — open the file in any audio player to hear it.

Run it:

    python examples/06_text_to_speech.py
    python examples/06_text_to_speech.py "Hello from the multimodal deep dive."
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

if not providers.supports("tts"):
    print(
        f"PROVIDER={providers.provider_name()} has no text-to-speech API.\n"
        f"Claude cannot synthesize speech. To run this example, set PROVIDER=openai\n"
        f"in .env. Skipping the call; nothing went wrong."
    )
    sys.exit(0)

text = sys.argv[1] if len(sys.argv) > 1 else (
    "Welcome to the multimodal deep dive. A model takes more than text — "
    "images and audio — and the skill is putting the right modality in the right slot."
)

print(f"Synthesizing speech for:\n  {text!r}\n")

audio_bytes = providers.speak(text, voice="alloy")

out_dir = os.path.join(ROOT, "out")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "spoken.mp3")
media.save_bytes(out_path, audio_bytes)

print(f"Wrote {os.path.relpath(out_path, ROOT)}  ({len(audio_bytes):,} bytes)")
print("Open it in any audio player to hear it.")

print(
    "\nText in, audio out. Combine 05 + 06 and you have a full voice loop: speak a "
    "question, transcribe it, answer it, speak the answer back."
)
