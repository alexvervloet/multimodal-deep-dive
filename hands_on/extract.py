"""
Capstone — `extract.py`: a screenshot -> structured-data CLI (with a voice mode).
=================================================================================

Everything in the repo, assembled into one tool you can actually use. It takes an
image of a document and returns clean JSON, choosing the schema from a small
built-in set or following one you describe. It also has an optional --voice mode
that speaks the result aloud (OpenAI only) and a --token-cost flag that prints the
image's estimated token cost (offline) before sending — so you see what you're
about to spend.

It's just the library wired to a CLI: providers.image_block + providers.chat for
the extraction, tokens.estimate for the cost, providers.speak for the voice.

Examples:

    # Extract a receipt to JSON (default schema):
    secrun python hands_on/extract.py assets/receipt.png

    # See the token cost first, and pretty-print:
    secrun python hands_on/extract.py assets/receipt.png --token-cost

    # Describe your own schema in plain English:
    secrun python hands_on/extract.py assets/chart.png --schema "a list of bars, each with its height as a number"

    # Speak a one-line summary of the result aloud (OpenAI only):
    secrun python hands_on/extract.py assets/receipt.png --voice

    # Save the JSON to a file:
    secrun python hands_on/extract.py assets/receipt.png -o out/receipt.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from multimodal import media, providers, tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A couple of ready-made schemas so the common cases need no --schema flag.
RECEIPT_SCHEMA = (
    '{ "merchant": string, "date": string, '
    '"items": [ { "name": string, "price": number } ], '
    '"subtotal": number, "tax": number, "total": number }'
)


def build_system(schema: str) -> str:
    return (
        "You extract structured data from an image. "
        "Return ONLY a JSON object, no markdown fences, no prose. "
        f"Use this shape (use null for anything missing): {schema}"
    )


def strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences a model sometimes adds despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn an image of a document into structured JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="path to the image to extract from")
    parser.add_argument(
        "--schema",
        default=None,
        help="describe the JSON shape you want (default: a receipt schema)",
    )
    parser.add_argument("-o", "--output", default=None, help="write the JSON to this file")
    parser.add_argument(
        "--token-cost",
        action="store_true",
        help="print the image's estimated token cost (offline) before sending",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="speak a one-line summary of the result aloud (OpenAI only)",
    )
    args = parser.parse_args()

    load_dotenv()
    providers.ensure_ready()

    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = os.path.join(ROOT, image_path)
    if not os.path.exists(image_path):
        sys.exit(f"No such image: {args.image}")

    data, media_type = media.load_bytes(image_path)
    print(f"Provider: {providers.describe()}")
    print(f"Image:    {os.path.relpath(image_path, ROOT)}  ({len(data):,} bytes)")

    # Optional: show the token cost offline, before spending anything.
    if args.token_cost and media_type == "image/png":
        w, h = media.png_size(data)
        est = tokens.estimate(providers.provider_name(), w, h)
        print(f"Est. image tokens ({est.provider}): {est.tokens}  ({est.explanation})")

    schema = args.schema or RECEIPT_SCHEMA
    print(f"Schema:   {schema}\n")

    raw = providers.chat(
        system=build_system(schema),
        content_blocks=[
            providers.text_block("Extract this image as JSON."),
            providers.image_block(data, media_type),
        ],
    )

    cleaned = strip_fences(raw)
    try:
        result = json.loads(cleaned)
        pretty = json.dumps(result, indent=2)
        print(pretty)
    except json.JSONDecodeError as e:
        print("Model did not return valid JSON:\n")
        print(raw)
        sys.exit(f"\n(json parse error: {e})")

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        media.save_bytes(out_path, pretty.encode("utf-8"))
        print(f"\nWrote {os.path.relpath(out_path, ROOT)}")

    # Optional: speak a short summary aloud. Degrades gracefully on Claude.
    if args.voice:
        summary = ", ".join(f"{k}: {v}" for k, v in result.items() if not isinstance(v, (list, dict)))
        summary = summary or "Extraction complete."
        if not providers.supports("tts"):
            print(
                f"\n[--voice] PROVIDER={providers.provider_name()} has no text-to-speech "
                f"API; skipping audio. Use PROVIDER=openai to hear it."
            )
        else:
            audio = providers.speak(f"Here is the extracted data. {summary}")
            out_dir = os.path.join(ROOT, "out")
            os.makedirs(out_dir, exist_ok=True)
            voice_path = os.path.join(out_dir, "extract_summary.mp3")
            media.save_bytes(voice_path, audio)
            print(f"\n[--voice] Wrote {os.path.relpath(voice_path, ROOT)} — open it to listen.")


if __name__ == "__main__":
    main()
