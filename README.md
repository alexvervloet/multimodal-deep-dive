# Multimodal: A Guided Deep Dive

A hands-on playground for learning how multimodal LLMs actually work, by feeding them more
than text. You'll send images and audio to a model from scratch and understand every moving
part. The content-block shape for images, structured extraction from a screenshot, comparing
multiple images, speech-to-text and text-to-speech, image generation, multimodal RAG, and
the surprisingly large token cost of an image. No framework magic, just enough code to see
how each modality rides into the model's context.

This repo is standalone and teaches everything it needs on its own. It builds naturally on
ideas from the sibling repos, the API calls
([OpenAI](https://github.com/alexvervloet/openai-api-deep-dive),
[Claude](https://github.com/alexvervloet/claude-api-deep-dive)) and especially
[RAG](https://github.com/alexvervloet/rag-deep-dive), where Section 9 here is RAG over
images. Its code depends on none of them.

Like its siblings, walk through it. Each section ends with something to run, and the first
two run offline and free. [EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for
each section.

---

## 0. The one big idea

> **A multimodal model takes images and audio in its context as well as text. The skill is
> putting the right modality in the right slot, and paying attention to what each one
> costs.**

That is the whole repo. A text request sends a string. A multimodal request sends a list of
typed content blocks: a text block and an image block, side by side in one user turn. Audio
gets its own slot, a transcription endpoint. Everything below, from extraction and
multi-image comparison to audio, generation, and multimodal RAG, is a variation on which
modality goes in which slot. And because an image is not free, since it gets tokenized by its
pixels, the second half of the skill is knowing what each slot costs. Hold onto that and
none of this feels complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Choose your provider (set PROVIDER in .env); your key loads separately
cp .env.example .env
#    Your API key does NOT go in .env. Store it in your OS keychain and run
#    lessons with `secrun`: 2-minute setup in ../docs/SECRETS.md.

# 4. Confirm everything is wired up (makes no API call, costs nothing)
secrun python check_setup.py       # secrun injects your key so the check can see it
```

Unlike the other repos in this series, the providers here genuinely differ in capability
rather than only in request shape, so the `PROVIDER` choice matters.

| `PROVIDER` | Vision (image in) | Audio (STT / TTS) | Image generation | Key needed |
|------------|:-----------------:|:-----------------:|:----------------:|------------|
| `openai` (default) | yes, `gpt-5.4-nano` | yes, Whisper and TTS | yes, `gpt-image-1` | `OPENAI_API_KEY` |
| `claude` | yes, `claude-haiku-4-5` | no native audio API | no, vision in only | `ANTHROPIC_API_KEY` |

Vision works on both. Audio and image generation are OpenAI-only, because Claude has no
native audio API and does not generate images. The single file that knows all of this is
[multimodal/providers.py](multimodal/providers.py). Where a feature is single-provider, the
example says so and skips the call with a clear message instead of crashing. The default is
`openai`, because it exercises every section.

> **Start before spending anything.** Example 01, an offline vision mock, and example 09,
> the image token math, run with no key and no cost. The rest make small, cheap calls.

---

## 2. Image input, and the content-block shape

A multimodal message is not a string. It is a list of typed content blocks. The foundational
move is building that list: a text block holding your question and an image block holding
your picture, in one user turn.

```bash
python examples/01_vision_offline.py        # offline, no key, no cost
```

This one is fully offline. It builds the real content-block list for your provider and
hands it to a tiny in-process mock vision model that reads the image's bytes, parsing the
PNG header, to prove the picture actually rode along. You see the exact request shape, and
that the image is base64 bytes in a block, before spending anything. The only thing that
differs across providers is the envelope around those bytes, which
[providers.image_block](multimodal/providers.py) builds. The rest of your code never
cares.

---

## 3. Vision: describe an image

Swap the mock for a real model and the same content-block list now gets a real
description back. Vision works on **both** providers, so this runs either way.

```bash
secrun python examples/02_vision_describe.py
secrun python examples/02_vision_describe.py assets/chart.png "How many bars are there?"
```

This is the hello world of multimodality. Hand the model a picture and a question in one
turn, get prose back. The image rode in the same message as the question, and that is the
slot. See [examples/02_vision_describe.py](examples/02_vision_describe.py). It's a dozen
lines around `providers.chat()`.

---

## 4. Document understanding, from screenshot to structured JSON

Describing an image is nice. Extracting it is useful. The real workhorse of business
multimodality is turning a picture of a document, a receipt, invoice, form, or screenshot,
into clean JSON your code can use.

```bash
secrun python examples/03_structured_extraction.py
```

The technique is pure prompting. Send the image, and in the system prompt demand a specific
JSON shape and "return ONLY JSON." The example then `json.loads` the reply to prove it is
real machine-usable data, and strips the ` ```json ` fences a model sometimes adds anyway.
The bundled `receipt.png` becomes `{merchant, date, items, subtotal, tax, total}`. That's a
screenshot-to-database pipeline in about 30 lines, and the capstone wraps exactly this into
a CLI.

---

## 5. Multiple images and comparison

Nothing says a user turn has only one image. Put several image blocks in one message and ask
the model to relate them. What changed? Do these match? The content-block list gets
longer.

```bash
secrun python examples/04_multiple_images.py
```

The example sends the receipt and the bar chart in one turn and asks the model to compare
them. A small prompt trick helps: label each image with a text block before it, "Image A:",
"Image B:", so the model keeps them straight. The slot holds more than one image, and the
model attends to all of them together. More images means more image tokens, and Section 10
does that math.

---

## 6. Audio in, or speech-to-text

A new modality, a new slot. Audio does not go in a chat content block. It goes to a
dedicated transcription endpoint, Whisper. You hand it audio bytes and get back text, which
can then flow into any text or vision prompt.

```bash
secrun python examples/05_transcribe_audio.py
secrun python examples/05_transcribe_audio.py path/to/your/voice.mp3
```

> **OpenAI-only.** Claude has no native audio API. With `PROVIDER=claude` this
> example explains that and exits cleanly. It does not crash.

The bundled `note.wav` is a self-made 440 Hz tone rather than real speech, because we don't
ship a recording of a person, so a perfect transcript is empty. The point is the request
shape and the round-trip. Point it at a real voice file to see real text.

---

## 7. Audio out, or text-to-speech

The mirror image of Section 6. Text in, audio out. Hand the TTS endpoint a string and a
voice, and get back audio bytes you save and play.

```bash
secrun python examples/06_text_to_speech.py
secrun python examples/06_text_to_speech.py "Hello from the multimodal deep dive."
```

> **OpenAI-only**, like transcription. Claude has no TTS API; the example skips
> cleanly on `claude`.

The result gets written to `out/spoken.mp3`, which is git-ignored. Open it in any audio
player. Combine Sections 6 and 7 and you have a full voice loop: speak a question,
transcribe it, answer it, speak the answer back.

---

## 8. Image generation and editing

So far every example put an image into the model. This one gets an image out. A text prompt
becomes a brand-new picture, via `gpt-image-1`.

```bash
secrun python examples/07_image_generation.py
secrun python examples/07_image_generation.py "a watercolor fox reading a book"
```

> **OpenAI-only, and this is the biggest capability gap in the repo. Claude
> does NOT generate images.** Claude is vision-**in** only: it can describe,
> compare, and extract from images you give it, but it cannot create one. There is
> no Anthropic image-generation endpoint. With `PROVIDER=claude` the example says
> so and exits cleanly.

The generated PNG gets saved to `out/generated.png`. Remember the asymmetry. Every provider
here can read images. Only OpenAI can write one. Pick your provider around what your app
actually needs.

---

## 9. Multimodal RAG, retrieving over images and text

RAG, from the [RAG deep dive](https://github.com/alexvervloet/rag-deep-dive), puts the right
text in the model's context. But what if your knowledge base is images: screenshots, scanned
pages, photos? You cannot embed a picture with a text embedder. The most practical pattern
is caption-then-embed.

```bash
secrun python examples/08_multimodal_rag.py
secrun python examples/08_multimodal_rag.py "which image has prices on it?"
```

1. **Caption** each image with a vision model (image → a sentence).  *[vision]*
2. **Index** those captions like any text and retrieve the closest.  *[retrieval]*
3. **Answer from the original image**: feed the actual picture back to the model.  *[vision]*

So vision bookends a text-retrieval core, and it runs on either provider. To stay
from-scratch and dependency-free, the example uses a tiny bag-of-words cosine in place of a
real embedding model. Not production-grade, and enough to show the architecture. Swap in
real embeddings, or true image embeddings, from the RAG dive and the shape is identical.

---

## 10. The token math of images

Here is the most surprising fact in multimodality. An image is not free, and it is not one
token. A model tokenizes it into many tokens based on its pixel dimensions, and a big
screenshot can cost more than a page of text.

```bash
python examples/09_image_token_math.py        # offline, no key, no cost
```

This computes that cost with pure arithmetic and no API call, using the real PNG dimensions
of the repo's assets and each provider's documented scheme. OpenAI uses a base plus 512×512
tiles, Claude uses roughly area divided by 750. The two numbers differ, and the stable
lesson is the shape. Tokens scale with pixels, so downscaling before you send is your
cheapest optimization. The example proves it by pricing a phone screenshot at full size and
at half size, where halving each side saves about 11k tokens. These are teaching
approximations. For billing, trust the `usage` field in the real response.

---

## 11. Native PDF, handing the model the document instead of a screenshot

Section 4 extracted a document by turning it into a picture and using vision. That is the
workaround everyone starts with. It throws away the real text and the page structure, and
it struggles past one page. What enterprise document pipelines actually reach for is native
PDF input. Pass the PDF bytes as their own content block and the model reads the document
itself.

```bash
secrun python examples/10_native_pdf.py
secrun python examples/10_native_pdf.py path/to/your.pdf
```

It is the same one big idea: the right modality in the right slot. A PDF is another slot.
`providers.pdf_block(bytes)` rides in the same user turn as your question, exactly like an
image block, and only the envelope differs per provider: OpenAI takes a `file` part, Claude
takes a `document` block. The example runs the same JSON-extraction discipline as §4 but
over the bundled `invoice.pdf`, a real document rather than a screenshot of one, and
`json.loads` the reply to prove it is machine-usable. Native PDF is the default for
document work. The §4 screenshot route is the fallback for a model that cannot take a PDF,
not the other way around. Native PDF support is model-specific, and the example exits
cleanly if the active model refuses it.

---

## The capstone: `extract.py`

Everything assembled into a CLI you can actually use. It takes an image of a document and
returns clean JSON, with the schema either a built-in default or something you describe in
plain English. It can print the image's token cost first, offline, and speak a summary of
the result aloud on OpenAI.

```bash
# Extract a receipt to JSON (default schema):
secrun python hands_on/extract.py assets/receipt.png

# See the token cost before sending, and pretty-print:
secrun python hands_on/extract.py assets/receipt.png --token-cost

# Describe your own schema in plain English:
secrun python hands_on/extract.py assets/chart.png --schema "a list of bars, each with a height number"

# Speak a summary aloud (OpenAI only; skips gracefully on claude):
secrun python hands_on/extract.py assets/receipt.png --voice

# Save the JSON to a file:
secrun python hands_on/extract.py assets/receipt.png -o out/receipt.json
```

Read [hands_on/extract.py](hands_on/extract.py). It's the library wired to a CLI:
`image_block` and `chat` for extraction, `tokens.estimate` for the cost, `speak` for the
voice. **Suggested exercise:** point it at your own screenshot with a `--schema` you
invent, whether an invoice, a form, or a nutrition label. When it returns JSON you can
`json.loads`, the screenshot-to-database idea has clicked.

---

## Which provider should I use?

Because capability differs here and not just request shape, provider choice is a real
decision. It comes straight from the one big idea. Every provider can put an image in a
slot. Not all of them fill every slot.

| What you need | Reach for | Why |
|---------------|-----------|-----|
| Read / describe / compare images | **Either** | Vision (image input) works on both providers |
| Extract structured data from screenshots | **Either** | It's a vision call + a JSON-shaped prompt, so provider-agnostic |
| Transcribe audio (speech-to-text) | **OpenAI** | Claude has no native audio API |
| Synthesize speech (text-to-speech) | **OpenAI** | Same: audio out is OpenAI-only |
| Generate or edit an image | **OpenAI** | Claude is vision-in only; it cannot create images |
| One key, the whole repo | **OpenAI** | It exercises every section; `claude` covers the vision half |

Rule of thumb. If your app is vision-only, doing analysis, extraction, and comparison,
either provider works and you can stay provider-agnostic. If it touches audio or image
generation, you need OpenAI, or a specialist audio or image vendor, for that piece, and you
can still use Claude for the vision parts. Don't pick a provider on vibes. Pick it on which
slots your app fills.

---

## Where to go next

You've fed a model every modality it accepts. What comes next is more of the same idea,
with more fidelity and more modalities.

- **Real-time and streaming audio.** Low-latency voice agents: turn detection, interruption
  (barge-in), and speech-to-speech against the transcribe, LLM, synthesize pipeline. The
  [Realtime Voice dive](https://github.com/alexvervloet/realtime-voice-deep-dive) builds a
  from-scratch simulator of exactly this.
- **Video.** Sampling frames as images, which is multi-image RAG over time, or true native
  video inputs as they roll out.
- **True image embeddings.** Embedding pictures directly, CLIP-style, instead of
  caption-then-embed, for retrieval that doesn't lose detail to a caption.
- **Structured outputs and JSON mode.** Provider features that guarantee valid JSON from a
  vision call, replacing the fence-stripping in Section 4.
- **Image editing and inpainting.** Masks and reference images rather than text-to-image
  alone.
- **Multi-page and scanned PDFs.** §11 does native PDF input on a one-page invoice. Real
  pipelines handle long, multi-page, and scanned documents, where you may still fall back
  to page-image vision, plus citations back to the source page.
- **Higher-resolution vision.** Newer models accept larger images at pixel-accurate
  coordinates, which is good for computer-use and dense screenshots, at higher token cost.

Every one of these sits on top of the idea you started with. The right modality in the
right slot, mindful of the cost.

---

## From teaching code to production

The "Where to go next" section is about adding modalities. This one is about the
operational layer any multimodal app needs once people rely on it. It is independent of
which modality you send, and the same for any LLM app.

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| Send images at their original size | **Downscale to a budget**: Section 10's tokens scale with pixels, so resize before sending and cap dimensions |
| Token cost is estimated offline (Section 10) | A **cost budget** enforced per request from the *actual* `usage`, since one big image can dwarf the text |
| `chat()` / `transcribe()` / `speak()` called bare | The calls wrapped in **retries + backoff** so a flaky provider doesn't fail the request |
| Extraction trusts the model's JSON (fence-stripping) | **Schema validation**, through structured outputs or a validator, so a malformed extraction gets caught instead of used |
| Uploaded images and audio are trusted input | **Guardrails**: user-supplied media is untrusted; images can carry injected text, audio can carry injected speech |
| One provider hard-wired per modality | A **capability router** that sends each modality to a provider that supports it, with fallbacks |
| Output (generated images, TTS) written to a local file | A **storage + CDN** path, with retention and access control on user media |

The general ops machinery (observability, cost, reliability, caching, guardrails, prompt
versioning, eval gates) gets built from scratch and wired into one running app in
[Production](https://github.com/alexvervloet/ai-in-production-deep-dive), which is #8 in
the series. It runs offline on a mock provider, so you can see the whole thing with no
key and no cost.

---

## File map

```
check_setup.py              ← run first: Python, packages, provider, key, assets, capabilities
README.md                   ← this guide
EXERCISES.md                ← predict-then-run prompts, one per section
multimodal/                 ← the from-scratch library (read it!)
  providers.py              ← the ONLY provider-specific file: vision chat + audio/image-gen
  media.py                  ← dependency-free load/save + PNG-size helpers
  tokens.py                 ← estimate an image's token cost offline (Section 10)
assets/                     ← tiny, self-made sample media (no downloads)
  make_assets.py            ← regenerates the assets with the standard library only
  receipt.png               ← a "receipt" for the extraction demo
  chart.png                 ← a bar chart for the multi-image demo
  note.wav                  ← a 1-second tone, a stand-in audio clip
  invoice.pdf               ← a one-page invoice PDF for the native-PDF demo
hands_on/
  extract.py                ← capstone: screenshot -> JSON CLI (+ token cost, + voice)
examples/
  01_vision_offline.py      ← the content-block shape, via an offline mock (no key)
  02_vision_describe.py     ← describe an image (real vision call; both providers)
  03_structured_extraction.py ← receipt image -> structured JSON
  04_multiple_images.py     ← two images in one turn; compare them
  05_transcribe_audio.py    ← speech-to-text (OpenAI; skips on claude)
  06_text_to_speech.py      ← text-to-speech (OpenAI; skips on claude)
  07_image_generation.py    ← text -> image (OpenAI; claude can't generate images)
  08_multimodal_rag.py      ← caption-then-embed RAG over images (both providers)
  09_image_token_math.py    ← how images become tokens (offline, no key)
  10_native_pdf.py          ← native PDF input: extract an invoice PDF to JSON (both providers)
```

(`out/` is created by the image-generation and text-to-speech examples and is
git-ignored.)

---

## Troubleshooting

Run `secrun python check_setup.py` first; it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `PROVIDER=... needs ... in the environment` | Set `PROVIDER` in `.env`, then load the key from your keychain by running under `secrun`. See [SECRETS.md](../docs/SECRETS.md). |
| `PROVIDER=claude has no speech-to-text / text-to-speech API` | Working as intended; Claude has no native audio. Use `PROVIDER=openai` for audio, or run the vision examples on `claude`. |
| `PROVIDER=claude cannot generate images` | Working as intended; Claude is vision-in only. Use `PROVIDER=openai` for image generation. |
| `ModuleNotFoundError` (openai / anthropic / rich) | Dependencies aren't installed or the venv isn't active. `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| Sample assets MISSING | Regenerate them offline: `python assets/make_assets.py`. |
| Extraction returns prose, not JSON / a parse error | The model didn't follow the JSON instruction. Tighten the schema in the prompt, or try a stronger model. The examples strip ` ```json ` fences for you. |
| The transcript is empty | The bundled `note.wav` is a tone, not speech, so there's nothing to transcribe. Point it at a real voice recording. |
| An image "costs" thousands of tokens | That's real: images are tokenized by pixels (Section 10). Downscale before sending. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained. Open it, read the docstring
at the top, and run it directly. [multimodal/providers.py](multimodal/providers.py)
is the whole story.

---

## The series

This is one of the standalone, hands-on deep dives into building with LLM APIs. Eight
core dives, plus the bonus ones listed below. Each one stands on its own, with its own
setup, examples, and capstone, and they all share one house style. Provider-agnostic
where it makes sense, built from scratch with no frameworks, offline-first examples, and
a real capstone at the end. Do them in any order. This sequence builds naturally.

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive): the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive): the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive): shape model behavior with better prompts, using zero-shot and few-shot, chain-of-thought, and roles
4. [RAG](https://github.com/alexvervloet/rag-deep-dive): answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive): measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive): give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive): attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive): operate one app end to end, across observability, cost, reliability, caching, guardrails, prompt versioning, and eval gates

**Bonus dives**, standalone and slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive): manage what's in the window, with memory, compaction, and assembly
- [AI Data Engineering](https://github.com/alexvervloet/ai-data-engineering-deep-dive): the corpus behind the index, with versions, lineage, ACLs, and deletes
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive): images and audio as well as text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive): teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive): serve tools, data, and prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive): run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive): build on the loop, adding hooks, permissions, sandboxing, and subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive): low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive): watch a running app over time, covering drift, quality, alerting, and the feedback loop
- [Architecture](https://github.com/alexvervloet/architecture-deep-dive): the seams between the components, each decision measured rather than asserted
- [GenAI Security](https://github.com/alexvervloet/genai-security-deep-dive): treat the model as an untrusted principal, and put identity, supply chain, isolation, budgets, and release gates around it
- [Inference Platform Engineering](https://github.com/alexvervloet/inference-platform-deep-dive): turn finite GPU memory and a request queue into latency, throughput, and a fleet size you can defend
- [Testing & Delivery](https://github.com/alexvervloet/testing-and-delivery-deep-dive): decide whether a build is fit to promote, using evidence, gates, staged rollout, and rollback
- [Professional Tools](https://github.com/alexvervloet/professional-tools-deep-dive): rebuild each hand-written piece with the tool professionals reach for, and measure both

And the whole series lands in one codebase in the
[capstone](https://github.com/alexvervloet/deep-dive-capstone): a codebase Q&A tool
built step by step, one tag per dive.

**Multimodal is a bonus dive in the series**; it slots most naturally after the two
API dives (#1–2) and pairs with RAG (#4), whose retrieval ideas Section 9 extends to
images.
