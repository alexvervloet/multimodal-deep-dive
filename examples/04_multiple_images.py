"""
Example 04: multiple images & comparison.

Nothing says a user turn has only one image. You can put SEVERAL image blocks in
one message and ask the model to compare them: "what changed?", "which is
cheaper?", "do these match?". The content-block list just gets longer.

Here we send the receipt AND the bar chart in the same turn and ask the model to
relate them. The lesson is that "the right slot" can hold more than one image, and
the model attends to all of them together. (Note: more images = more image tokens
and example 09 shows exactly how much.)

Vision works on both providers, so this runs either way.

Run it:

    secrun python examples/04_multiple_images.py
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

# Load both images.
receipt, receipt_type = media.load_bytes(os.path.join(ROOT, "assets", "receipt.png"))
chart, chart_type = media.load_bytes(os.path.join(ROOT, "assets", "chart.png"))

# One user turn, THREE blocks: a question and two labeled images. Interleaving a
# little text before each image ("Image A:", "Image B:") helps the model keep them
# straight: a small but real prompt-engineering trick for multi-image turns.
content = [
    providers.text_block(
        "I'm sending two images. Image A is a cafe receipt; Image B is a bar chart. "
        "First, briefly say what each one shows. Then tell me: do they appear to be "
        "about the same thing, or unrelated?"
    ),
    providers.text_block("Image A:"),
    providers.image_block(receipt, receipt_type),
    providers.text_block("Image B:"),
    providers.image_block(chart, chart_type),
]

print(f"Sending 2 images in one turn ({len(receipt):,} + {len(chart):,} bytes)\n")

answer = providers.chat(
    system="You compare images carefully and answer in a short, structured way.",
    content_blocks=content,
)

print("Model:")
print(answer)

print(
    "\nTwo images, one slot, one answer. The list of blocks is the only thing that "
    "grew. Sending more images costs more tokens; example 09 does that math."
)
