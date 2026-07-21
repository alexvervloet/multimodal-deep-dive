"""
Example 02: describe an image (real vision call).

Same content-block shape as example 01, a text block plus an image block, but
now it goes to a real vision model (gpt-4o-mini or claude-haiku-4-5). Vision works
on BOTH providers, so this runs either way; the only difference is the image
block's shape, which providers.image_block() handles for you.

This is the "hello world" of multimodality: hand the model a picture and a
question, get a description back.

Run it:

    secrun python examples/02_vision_describe.py
    secrun python examples/02_vision_describe.py assets/chart.png "How many bars are there?"
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

# Optional CLI args: an image path and a question.
image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "receipt.png")
question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in two sentences."

image_data, media_type = media.load_bytes(image_path)

# The whole call: one text block + one image block, in one user turn.
content = [
    providers.text_block(question),
    providers.image_block(image_data, media_type),
]

print(f"Image:    {os.path.relpath(image_path, ROOT)}  ({len(image_data):,} bytes)")
print(f"Question: {question}\n")

answer = providers.chat(
    system="You are a careful assistant that describes images accurately and concisely.",
    content_blocks=content,
)

print("Model:")
print(answer)

print(
    "\nThe image rode in the SAME message as the question. That's the slot. "
    "Next (example 03), we ask for STRUCTURED data instead of prose."
)
