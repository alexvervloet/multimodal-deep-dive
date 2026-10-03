"""
Example 09: the token math of images (offline, no key).

Images are not free, and they are not one token. A model *tokenizes* an image
based on its pixel dimensions, and a big screenshot can cost more than a page of
text. This example computes that cost with pure arithmetic: no API call, no key,
no cost, so you can budget BEFORE you send.

It uses the real PNG dimensions of the repo's assets (read straight from the file
header) and runs them through each provider's documented tokenization scheme:

  OpenAI (gpt-6-luna): about 1.2 tokens per 32x32 patch, after a resize that
      depends on `detail`. Measured against real bills; see tests/test_tokens.py.
  Claude: tokens ≈ (width * height) / 750, with a cap.

The two numbers differ, which is expected; the providers tokenize differently. The
stable lesson is the SHAPE of the cost: tokens scale with pixels, so the single
biggest lever you have is **downscaling the image before you send it**. We prove
that by also pricing a half-size copy.

The second OpenAI column is the trap. Leave `detail` out and luna uses "auto",
which on this model means "don't shrink": the 4K grab costs more than three times
what "high" charges. The repo's providers.py sets "high" for exactly this reason.

  ACCURACY: These are teaching approximations. The real token count always
      comes back in the API response's usage field; trust that for billing.

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
        auto = tokens.openai_image_tokens(w, h, detail="auto")
        c = tokens.estimate("claude", w, h)
        yield (label, f"{w}x{h}", str(o.tokens), str(auto.tokens), str(c.tokens))


def main() -> None:
    print("How many tokens does an image cost? (computed offline, no key)\n")

    if _RICH:
        table = Table(title="Estimated image-input tokens")  # type: ignore[possibly-undefined]
        table.add_column("image", style="cyan")
        table.add_column("size", justify="right")
        table.add_column("openai high", justify="right", style="green")
        table.add_column("openai auto", justify="right", style="red")
        table.add_column("claude", justify="right", style="magenta")
        for r in rows():
            table.add_row(*r)
        Console().print(table)  # type: ignore[possibly-undefined]
    else:
        print(f"{'image':<22}{'size':>12}{'openai high':>13}{'openai auto':>13}{'claude':>10}")
        for label, size, o, auto, c in rows():
            print(f"{label:<22}{size:>12}{o:>13}{auto:>13}{c:>10}")

    # The downscaling lever, made concrete: use a LARGE image, where it bites.
    # (A tiny image is only a few patches, so resizing it barely matters; the
    # lever bites once an image runs to thousands of patches.)
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
        "  - A big screenshot can cost thousands of tokens, more than a page of text.\n"
        "  - Tokens scale with pixels, so resizing down is your cheapest optimization.\n"
        "  - Set `detail` yourself. A default that changed between models just\n"
        "    tripled the cost of a 4K image.\n"
        "  - The two providers tokenize differently; the SHAPE of the cost is the\n"
        "    stable lesson. For billing, trust the usage field in the real response."
    )


if __name__ == "__main__":
    main()
