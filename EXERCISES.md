# Exercises — make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts.

How to use it: work the section first, then come back. **Commit to an answer
before you run or reveal** — the prediction is where the learning happens. Answers
are hidden behind ▸ toggles.

> Examples 01 (offline vision mock) and 09 (image token math) are **(offline)** —
> no API call, no cost. The rest make small, cheap calls; the audio and
> image-generation ones are OpenAI-only.

---

## Section 2 — What a multimodal message is **(offline)**

**Recall.** A text-only request sends a string. What does a multimodal request
send instead, and where does the image go relative to the question?

<details><summary>▸ Answer</summary>

A *list of typed content blocks* — a text block and an image block — inside **one**
user turn. The image rides in the same message as the question. That list is the
"put the right modality in the right slot" idea, literally.
</details>

**Do (offline).** In `examples/01_vision_offline.py`, the image block looks
different on `openai` vs `claude`. What's actually the same in both, and what's
just the envelope?

<details><summary>▸ Answer</summary>

The same thing in both: the image is **base64-encoded bytes**. Only the wrapper
differs — OpenAI uses `image_url` with a `data:` URI; Claude uses an `image` block
with a typed `source`. `providers.image_block()` builds the right envelope so your
code never has to care.
</details>

---

## Section 3 — Vision: describe an image

**Predict, then run.** In `examples/02_vision_describe.py` you send a receipt and
ask "describe this image." Does this work on `claude` as well as `openai`? Why?

<details><summary>▸ Answer</summary>

Yes — vision (image **input**) works on **both** providers. It's the one
multimodal capability they share. Only the image-block shape differs, and the
provider shim hides that. (Audio and image *generation* are where they diverge.)
</details>

---

## Section 4 — Structured extraction

**Recall.** What single thing turns "describe this image" into a reliable
screenshot-to-JSON pipeline?

<details><summary>▸ Answer</summary>

The **prompt** — specifically, spelling out the exact JSON shape in the system
prompt and demanding "return ONLY JSON." You're not using a special endpoint; you're
constraining a normal vision call to emit machine-readable data, then `json.loads`
it.
</details>

**Do.** Run `examples/03_structured_extraction.py`. The model sometimes wraps its
JSON in ` ```json ` fences even when told not to. Why does the example strip
fences before parsing?

<details><summary>▸ Answer</summary>

Because a fenced block isn't valid JSON — `json.loads` would throw on the
backticks. Stripping fences is a small but necessary bit of defensive parsing for
any "give me JSON" prompt. A stricter approach is a structured-output / JSON-mode
feature, but fence-stripping works everywhere.
</details>

---

## Section 5 — Multiple images

**Predict.** In `examples/04_multiple_images.py` you send two images in one turn.
How does the request change compared to sending one?

<details><summary>▸ Answer</summary>

The content-block **list just gets longer** — two image blocks instead of one,
plus a little labeling text ("Image A:", "Image B:") so the model keeps them
straight. The slot can hold many images; the model attends to all of them
together. (More images = more image tokens — see Section 10.)
</details>

---

## Section 6 — Audio in (transcription)

**Recall.** Does audio go in a chat content block like an image does? And which
providers can transcribe?

<details><summary>▸ Answer</summary>

No — audio goes to a **dedicated transcription endpoint** (Whisper), not into a
chat block. You get text back, which can then flow into any text/vision prompt.
And it's **OpenAI-only** — Claude has no native audio API, so the example detects
that and skips cleanly on `claude`.
</details>

**Do.** Run `examples/05_transcribe_audio.py` on the bundled `note.wav`. Why is the
transcript empty (or near-empty), and is that a bug?

<details><summary>▸ Answer</summary>

Not a bug — `note.wav` is a self-made 440 Hz **tone**, not speech (we don't ship a
recording of a real person). There are no words to transcribe. The example exists
to show the request shape and round-trip; point it at a real voice file to see
real text.
</details>

---

## Section 7 — Audio out (text-to-speech)

**Recall.** Section 6 was audio→text. What's Section 7, and which providers can do
it?

<details><summary>▸ Answer</summary>

Text→audio: hand a string + a voice to the TTS endpoint, get audio bytes back
(saved to `out/spoken.mp3`). Also **OpenAI-only** — Claude has no TTS. Combine
Sections 6 and 7 and you have a full voice loop.
</details>

---

## Section 8 — Image generation

**Predict.** You set `PROVIDER=claude` and run
`examples/07_image_generation.py`. What happens?

<details><summary>▸ Answer</summary>

It prints a clear explanation that **Claude cannot generate images** — it's
vision-**in** only, with no image-generation endpoint at all — and exits cleanly
(no crash). Image generation is the single biggest capability gap in the repo:
both providers READ images, only OpenAI WRITES one.
</details>

---

## Section 9 — Multimodal RAG

**Recall.** You can't embed a picture with a text embedder. What's the
caption-then-embed pattern, in three steps?

<details><summary>▸ Answer</summary>

1. **Caption** each image with a vision model (image → a sentence).
2. **Index** those captions like any text (embed + retrieve).
3. At query time, retrieve the best caption, then **answer from the original
   image** by feeding the real picture back to the model.

Vision bookends a text-retrieval core. (`examples/08_multimodal_rag.py` uses a
from-scratch bag-of-words cosine in place of real embeddings to keep it
dependency-free — the architecture is the lesson.)
</details>

---

## Section 10 — The token math of images **(offline)**

**Predict, then run.** Before running `examples/09_image_token_math.py`: is an
image one token? Roughly how does its cost scale?

<details><summary>▸ Answer</summary>

No — an image is *tokenized* into many tokens, and the count scales with **pixel
dimensions**. A big screenshot can cost thousands of tokens (more than a page of
text). The two providers tokenize differently, but the shape is the same.
</details>

**Do (offline).** The example shows a phone screenshot costing far more than the
tiny assets. What's the single cheapest optimization, and why does halving a
*small* image change nothing?

<details><summary>▸ Answer</summary>

**Downscale before sending** — fewer pixels, fewer tokens. Halving a small image
changes nothing because it already fits in a single 512×512 tile (OpenAI) or is
already under the cap (Claude); the lever only bites once an image spans multiple
tiles, which the phone-screenshot row demonstrates (halving each side saved
~11k tokens).
</details>

---

## Capstone — `extract.py`

**Do.** Run `python hands_on/extract.py assets/receipt.png --token-cost`. You've
now combined three things in one command: the offline token estimate, the vision
call, and JSON extraction. Then add `--voice` (on `openai`) to hear the result —
and on `claude`, watch it skip the audio gracefully.

**Stretch.** Point `extract.py` at your OWN screenshot with a `--schema` you
describe in plain English (e.g. a form, an invoice, a nutrition label). When it
returns clean JSON you can `json.loads`, the screenshot-to-database idea has
clicked.

---

### Where to take it next

Invent your own multimodal pipeline. Chain Section 6 (transcribe) → a text/vision
prompt → Section 7 (speak) into a voice assistant, or wire Section 9's
caption-then-embed over a folder of your own screenshots. The first time you put a
non-text modality in the right slot and get something useful back, the idea has
landed.
