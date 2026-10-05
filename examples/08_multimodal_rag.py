"""
Example 08: multimodal RAG: retrieve over images + text.

RAG (from the RAG deep dive) puts the right text in the model's context. But what
if your knowledge base is IMAGES: screenshots, scanned pages, photos? You can't
embed a picture with a text embedder. The most practical pattern is
**caption-then-embed**:

  1. Caption each image with a vision model (image -> a sentence of text).   [vision]
  2. Index those captions like any other text.                              [retrieval]
  3. At query time, retrieve the best-matching caption, then ANSWER from the
     original image, feeding the actual picture back to the model.         [vision]

So vision bookends a text-retrieval core. Vision works on both providers, so this
runs either way. To stay from-scratch and dependency-free, the "embedding" here is
a tiny bag-of-words cosine: not a real embedding model, but enough to show the
*architecture*. (Swap in real embeddings from the RAG dive for production.)

Word overlap has one big weakness, and this example shows it on purpose: it only
matches words, not meaning. Two things keep it honest here. A stopword list stops
"the" and "of" from deciding the match (before it existed, the default question
retrieved the chart every time). And the captions are written for search: what
kind of image it is and what information it holds, in plain words. Measured
2026-10-05, five runs per query, both providers retrieved the right image for all
three questions below, 30 of 30. (One earlier miss came from the word "image"
itself, once the captions started saying it; it's a stopword here now.) But look at the scores on the claude stack: Haiku
captions the receipt as "itemized purchases", which shares no word with "prices
of items", so both images score 0 and the receipt wins only because it's listed
first. The example says so when it happens. Real embeddings know "itemized" and
"items" are related; that's the reason to use them.

Run it:

    secrun python examples/08_multimodal_rag.py
    secrun python examples/08_multimodal_rag.py "which image has prices on it?"
    secrun python examples/08_multimodal_rag.py "which bar is the tallest?"
"""

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from multimodal import media, providers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
providers.ensure_ready()
print(f"Provider: {providers.describe()}\n")

# Our tiny image "corpus": the two assets we ship.
CORPUS = ["receipt.png", "chart.png"]


# --- A from-scratch bag-of-words vectorizer + cosine (no embedding API). ----
# Words that carry no meaning. Without this list, "the" and "of" decided which
# image the default query retrieved, and they picked the wrong one. "image" and
# friends are stopwords too in a corpus where every caption describes an image.
STOPWORDS = set(
    "a an and are at be by can do does for from has have how i in is it its of on "
    "or that the this to what where which with you your "
    "image images picture photo shows showing".split()
)


def vectorize(text: str) -> dict[str, float]:
    """Map text to a term-frequency vector (a dict). Crude but transparent."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    vec: dict[str, float] = {}
    for w in words:
        vec[w] = vec.get(w, 0.0) + 1.0
    return vec


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two term-frequency vectors."""
    shared = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# --- Step 1: caption every image with the vision model. ---------------------
print("Indexing images (caption-then-embed)...")
index = []  # list of (filename, caption, vector, raw_bytes, media_type)
for name in CORPUS:
    data, media_type = media.load_bytes(os.path.join(ROOT, "assets", name))
    # A caption for RETRIEVAL, not for reading: say what kind of thing the image
    # is and what information it holds, in the words someone searching would use.
    # Word-overlap search only finds "prices" if the caption says "prices"; a
    # caption that only says "a latte for $4.50" can't match that query.
    caption = providers.chat(
        system="You write a single, information-dense sentence describing an image "
        "for a search index. Start with what kind of image it is (for example a "
        "receipt, a chart, a screenshot), then name the kinds of information it "
        "contains in plain words someone might search for (for example prices, "
        "totals, dates, names, measurements, trends), then any visible text or numbers.",
        content_blocks=[
            providers.text_block("Caption this image in one sentence."),
            providers.image_block(data, media_type),
        ],
    ).strip()
    index.append((name, caption, vectorize(caption), data, media_type))
    print(f"  {name}: {caption}")

# --- Step 2 & 3: retrieve the best caption, then answer from the real image. -
query = sys.argv[1] if len(sys.argv) > 1 else "where can I see the prices of items?"
print(f"\nQuery: {query!r}")

qvec = vectorize(query)
scores = [(cosine(qvec, row[2]), row) for row in index]
for score, row in sorted(scores, key=lambda pair: pair[0], reverse=True):
    print(f"  score {score:.2f}  {row[0]}")
ranked = [row for _, row in sorted(scores, key=lambda pair: pair[0], reverse=True)]
best_name, best_caption, _, best_data, best_type = ranked[0]
print(f"Retrieved: {best_name}  (caption: {best_caption})\n")
if scores and max(score for score, _ in scores) == 0:
    print("No caption shares a word with the query, so that pick was arbitrary.\n")

# Now answer GROUNDED in the actual image, not just its caption: the caption is
# how we find the picture; the picture has the detail.
answer = providers.chat(
    system="Answer the user's question using ONLY the provided image. "
    "If the image doesn't contain the answer, say so.",
    content_blocks=[
        providers.text_block(query),
        providers.image_block(best_data, best_type),
    ],
)

print("Answer (grounded in the retrieved image):")
print(answer)

print(
    "\nThat's multimodal RAG in miniature: vision to caption, text retrieval to "
    "find, vision again to answer. Production swaps the bag-of-words for real "
    "embeddings (or true image embeddings); the architecture is identical."
)
