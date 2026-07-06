"""
Example 03 — extract STRUCTURED data from an image (receipt -> JSON).
=====================================================================

Describing an image is nice; extracting it is useful. The real workhorse of
business multimodality is turning a picture of a document — a receipt, an
invoice, a form, a screenshot — into clean JSON your code can use.

The technique is just prompting: send the image, and in the system prompt demand
a specific JSON shape and "return ONLY JSON." Vision works on both providers, so
this runs either way. We parse the reply with json.loads to prove it's real,
machine-usable data, not prose.

Run it:

    secrun python examples/03_structured_extraction.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from multimodal import media, providers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
providers.ensure_ready()
print(f"Provider: {providers.describe()}\n")

# The schema we want back. Spelling it out in the prompt is what makes extraction
# reliable — the model fills these fields rather than free-styling.
SYSTEM = """You extract structured data from images of receipts.
Return ONLY a JSON object, no markdown, no prose, with exactly this shape:
{
  "merchant": string,
  "date": string,
  "items": [ { "name": string, "price": number } ],
  "subtotal": number,
  "tax": number,
  "total": number
}
If a field is missing from the receipt, use null."""

image_data, media_type = media.load_bytes(os.path.join(ROOT, "assets", "receipt.png"))

content = [
    providers.text_block("Extract this receipt as JSON."),
    providers.image_block(image_data, media_type),
]

raw = providers.chat(SYSTEM, content)

print("Raw model output:")
print(raw)

# Models sometimes wrap JSON in ```json fences despite instructions — strip them.
cleaned = raw.strip()
if cleaned.startswith("```"):
    cleaned = cleaned.split("```", 2)[1]
    if cleaned.startswith("json"):
        cleaned = cleaned[4:]
    cleaned = cleaned.strip()

print("\nParsed as real data (json.loads):")
try:
    data = json.loads(cleaned)
    print(json.dumps(data, indent=2))
    # Prove it's usable: do arithmetic the model never saw as a number.
    if isinstance(data.get("items"), list):
        n = len(data["items"])
        print(f"\nYour code can now use it directly — e.g. {n} line item(s) found.")
except json.JSONDecodeError as e:
    print(f"Could not parse JSON ({e}). Tighten the prompt or try a stronger model.")

print(
    "\nThat's a screenshot-to-database pipeline in ~30 lines. The capstone wraps "
    "exactly this into a reusable CLI."
)
