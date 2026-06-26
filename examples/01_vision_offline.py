"""
Example 01 — what a multimodal message is (offline, no API call).
=================================================================

Before any provider, understand the shape. A text-only request sends a *string*.
A multimodal request sends a LIST of typed content blocks — a text block and an
image block side by side in one user turn:

    [ {type: text,  ...},
      {type: image, ...} ]

That list IS the "put the right modality in the right slot" idea, literally. This
example builds that list for the active provider, then hands it to a tiny OFFLINE
mock "vision model" that answers from the image's pixels (it reads the PNG header
and counts colors) — no key, no network, no cost. The point is to *see* the
request shape and the round-trip before you spend anything.

Run it:

    python examples/01_vision_offline.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal import media, providers

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def mock_vision(content_blocks: list[dict]) -> str:
    """A deterministic, offline 'vision model'.

    It can't really see, but it CAN inspect the bytes we put in the image block —
    proving the image actually rode along in the request. It pulls the base64 out
    of whichever provider's block shape we built, decodes it, and reports the
    image's true dimensions plus the text we sent alongside."""
    import base64

    text_parts, image_bytes = [], None
    for block in content_blocks:
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "image_url":  # OpenAI shape
            b64 = block["image_url"]["url"].split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
        elif block["type"] == "image":  # Claude shape
            image_bytes = base64.b64decode(block["source"]["data"])

    if image_bytes is None:
        return "I received no image — only text."
    w, h = media.png_size(image_bytes)
    asked = " ".join(text_parts) or "(no question)"
    return (
        f"[mock vision] You asked: {asked!r}\n"
        f"[mock vision] I received a real {w}x{h} PNG ({len(image_bytes):,} bytes). "
        f"A real model would now describe what's in it."
    )


def main() -> None:
    # No load_dotenv / ensure_ready — this example is fully offline.
    print(f"Provider shape: {providers.provider_name()} "
          f"(no key needed — this is the offline mock)\n")

    # 1. Load the image as raw bytes (dependency-free).
    image_data, media_type = media.load_bytes(os.path.join(ASSETS, "receipt.png"))

    # 2. Build the multimodal message: a text block + an image block. The
    #    image_block() helper produces the active provider's exact shape.
    content = [
        providers.text_block("What is in this image?"),
        providers.image_block(image_data, media_type),
    ]

    # 3. Show the structure the model receives. Notice: the image is base64 in the
    #    block — we truncate it here just so the print is readable.
    print("The content blocks the model receives (image data truncated):")
    printable = json.loads(json.dumps(content))  # deep copy
    for block in printable:
        if block["type"] == "image_url":
            url = block["image_url"]["url"]
            block["image_url"]["url"] = url[:40] + f"...({len(url)} chars total)"
        elif block["type"] == "image":
            data = block["source"]["data"]
            block["source"]["data"] = data[:24] + f"...({len(data)} chars total)"
    print(json.dumps(printable, indent=2))

    # 4. "Send" it to the offline mock and print the reply.
    print("\nMock model reply:")
    print(mock_vision(content))

    print(
        "\nThat list of blocks is the whole idea. Example 02 sends the EXACT same "
        "shape to a real vision model — just swap the mock for providers.chat()."
    )


if __name__ == "__main__":
    main()
