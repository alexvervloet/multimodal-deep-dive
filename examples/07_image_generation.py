"""
Example 07: image GENERATION (text -> image).

So far every example put an image INTO the model. This one gets an image OUT: a
text prompt becomes a brand-new picture (OpenAI's gpt-image-1).

  PROVIDER SUPPORT: OpenAI-only, and this is the biggest capability gap in
      the whole repo. **Claude does NOT generate images.** Claude is vision-IN
      only: it can describe, compare, and extract from images you give it, but it
      cannot create one. With PROVIDER=claude this example explains that and exits
      cleanly. Don't go looking for an Anthropic image-generation endpoint; there
      isn't one.

We save the generated PNG to out/generated.png (git-ignored).

Run it:

    secrun python examples/07_image_generation.py
    secrun python examples/07_image_generation.py "a watercolor fox reading a book"
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

if not providers.supports("image_gen"):
    print(
        f"PROVIDER={providers.provider_name()} cannot generate images.\n"
        f"Claude is vision-IN only and has no image-generation endpoint at all.\n"
        f"To run this example, set PROVIDER=openai in .env (gpt-image-1).\n"
        f"Skipping the call; nothing went wrong."
    )
    sys.exit(0)

prompt = sys.argv[1] if len(sys.argv) > 1 else (
    "A simple, friendly flat-design illustration of a robot looking at a photograph, "
    "soft colors, plenty of negative space."
)

print(f"Generating an image for:\n  {prompt!r}\n")
print("(This is the slowest and most expensive call in the repo: one image.)")

png_bytes = providers.generate_image(prompt, size="1024x1024")

out_dir = os.path.join(ROOT, "out")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "generated.png")
media.save_bytes(out_path, png_bytes)

w, h = media.png_size(png_bytes)
print(f"\nWrote {os.path.relpath(out_path, ROOT)}  ({w}x{h}, {len(png_bytes):,} bytes)")
print("Open it to see what came back.")

print(
    "\nRemember the asymmetry: every provider here can READ images, but only "
    "OpenAI can WRITE one. Pick your provider around what your app actually needs."
)
