"""
Example 09 — the token math of images (offline, no key).
========================================================

Images are not free, and they are not one token. A model *tokenizes* an image
based on its pixel dimensions, and a big screenshot can cost more than a page of
text. This example computes that cost with pure arithmetic — no API call, no key,
no cost — so you can budget BEFORE you send.

It uses the real PNG dimensions of the repo's assets (read straight from the file
header) and runs them through each provider's documented tokenization scheme:

  OpenAI (gpt-4o family): base tokens + 512x512 tiles after a resize.
  Claude: tokens ≈ (width * height) / 750, with a cap.

The two numbers differ — that's expected; the providers tokenize differently. The
stable lesson is the SHAPE of the cost: tokens scale with pixels, so the single
biggest lever you have is **downscaling the image before you send it**. We prove
that by also pricing a half-size copy.

  ⚠️  These are teaching approximations. The real token count always comes back
      in the API response's usage field — trust that for billing.

Run it:

    python examples/09_image_token_math.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False

from multimodal import media, tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read the true dimensions of our assets (no key, no image library).
ASSETS = []
for name in ("receipt.png", "chart.png"):
    data, _ = media.load_bytes(os.path.join(ROOT, "assets", name))
    w, h = media.png_size(data)
    ASSETS.append((name, w, h))

# Plus a couple of hypothetical sizes so you can see how cost scales.
HYPOTHETICAL = [
    ("a phone screenshot", 1170, 2532),
    ("a 4K screen grab", 3840, 2160),
]


def rows():
    for label, w, h in ASSETS + HYPOTHETICAL:
        o = tokens.estimate("openai", w, h)
        c = tokens.estimate("claude", w, h)
        yield (label, f"{w}x{h}", str(o.tokens), str(c.tokens))


def main() -> None:
    print("How many tokens does an image cost? (computed offline, no key)\n")

    if _RICH:
        table = Table(title="Estimated image-input tokens")  # type: ignore[possibly-undefined]
        table.add_column("image", style="cyan")
        table.add_column("size", justify="right")
        table.add_column("openai (gpt-4o-mini)", justify="right", style="green")
        table.add_column("claude", justify="right", style="magenta")
        for r in rows():
            table.add_row(*r)
        Console().print(table)  # type: ignore[possibly-undefined]
    else:
        print(f"{'image':<22}{'size':>12}{'openai':>10}{'claude':>10}")
        for label, size, o, c in rows():
            print(f"{label:<22}{size:>12}{o:>10}{c:>10}")

    # The downscaling lever, made concrete — use a LARGE image, where it bites.
    # (A tiny image already fits in one tile, so resizing it changes nothing; the
    # lever only matters once an image spans multiple tiles.)
    name, w, h = "a phone screenshot", 1170, 2532
    full = tokens.estimate("openai", w, h)
    half = tokens.estimate("openai", w // 2, h // 2)
    saved = full.tokens - half.tokens
    print(
        f"\nThe downscaling lever ({name}, openai):\n"
        f"  full {w}x{h}: {full.tokens} tokens  ({full.explanation})\n"
        f"  half {w // 2}x{h // 2}: {half.tokens} tokens  ({half.explanation})\n"
        f"  -> halving each side saved {saved} tokens on a single image."
    )

    print(
        "\nTakeaways:\n"
        "  - A big screenshot can cost thousands of tokens — more than a page of text.\n"
        "  - Tokens scale with pixels, so resizing down is your cheapest optimization.\n"
        "  - The two providers tokenize differently; the SHAPE of the cost is the\n"
        "    stable lesson. For billing, trust the usage field in the real response."
    )


if __name__ == "__main__":
    main()
