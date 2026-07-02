"""
Example 10 — native PDF input: hand the model the document, not a screenshot.
=============================================================================

Section 4 extracted a receipt by turning a document into a *picture* and using
vision. That's the workaround, and it's everywhere — but it throws away the real
text, the page structure, and struggles past one page. The tool enterprise
document pipelines actually reach for is **native PDF input**: you pass the PDF
bytes as their own content block and the model reads the document itself.

It's the same move as every other section — the right modality in the right slot.
A PDF is just another slot: `pdf_block(bytes)` rides in the same user turn as your
question, exactly like an image block. Both providers support it; only the
envelope differs (`providers.pdf_block` builds the right one).

We extract the bundled one-page `invoice.pdf` to JSON — the same discipline as §4,
but from a real document instead of a screenshot of one — and json.loads the reply
to prove it's machine-usable. If the active model rejects PDFs, it says so and
exits cleanly.

Run it:

    python examples/10_native_pdf.py
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

SYSTEM = """You extract structured data from invoice PDFs.
Return ONLY a JSON object, no markdown, no prose, with exactly this shape:
{
  "vendor": string,
  "invoice_number": string,
  "date": string,
  "line_items": [ { "description": string, "qty": number, "amount": number } ],
  "subtotal": number,
  "tax": number,
  "total": number
}
If a field is missing from the invoice, use null."""

pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "invoice.pdf")
pdf_data, _ = media.load_bytes(pdf_path)
print(f"Document: {os.path.basename(pdf_path)} ({len(pdf_data):,} bytes) — the PDF itself, not an image of it.\n")

content = [
    providers.text_block("Extract this invoice as JSON."),
    providers.pdf_block(pdf_data, filename=os.path.basename(pdf_path)),
]

try:
    raw = providers.chat(SYSTEM, content)
except Exception as e:  # noqa: BLE001 — degrade gracefully like the audio examples
    print("This model wouldn't accept the native PDF block:")
    print(f"  ({type(e).__name__}: {e})\n")
    print(
        "Native PDF support is model-specific. The lesson is the shape: a document is\n"
        "its own content slot — `pdf_block(bytes)` beside your question — so the model\n"
        "reads the real text and page structure instead of a screenshot's pixels. When\n"
        "a model doesn't take PDFs, the §4 screenshot route is the fallback, not the goal."
    )
    sys.exit(0)

print("Raw model output:")
print(raw)

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
    if isinstance(data.get("line_items"), list):
        print(f"\nYour code can use it directly — e.g. {len(data['line_items'])} line item(s), "
              f"total {data.get('total')}.")
except json.JSONDecodeError as e:
    print(f"Could not parse JSON ({e}). Tighten the prompt or try a stronger model.")

print(
    "\nSame extraction discipline as §4, but from the document itself. Native PDF is\n"
    "the enterprise default: real text (not OCR'd pixels), page structure preserved,\n"
    "and many pages in one call. Screenshot-to-vision is the fallback for when a\n"
    "model can't take a PDF — not the other way around."
)
