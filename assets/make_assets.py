"""
make_assets.py — generate the repo's sample assets, from scratch, offline.
=========================================================================

    python assets/make_assets.py

Every asset this repo ships is generated here with ONLY the Python standard
library — no Pillow, no downloads, no copyrighted images. That keeps the repo
self-contained and the assets tiny and self-made:

  receipt.png   — a small "receipt" image (the §3 structured-extraction demo)
  chart.png     — a tiny bar chart (the §4 multi-image comparison demo)
  note.wav      — one second of a 440 Hz tone (a stand-in audio clip for §5)
  invoice.pdf   — a one-page invoice PDF (the §11 native-PDF-input demo)

PNG is written by hand (zlib + a CRC) so you can *see* that an image is just
bytes; WAV uses the stdlib `wave` module. You normally never need to run this —
the assets are committed — but it's here so nothing in the repo is a mystery.
"""

import math
import os
import struct
import wave
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# A minimal PNG writer (truecolor RGB, no compression cleverness).
# A PNG is: an 8-byte signature, then chunks (IHDR, IDAT, IEND). Each chunk is
# length + type + data + CRC32. The pixels live in IDAT, zlib-compressed, with a
# filter byte (0 = "none") prepended to each row. That's the whole format.
# ---------------------------------------------------------------------------
def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: str, pixels: list[list[tuple[int, int, int]]]) -> None:
    """Write `pixels` (a 2-D list of (r,g,b)) to `path` as a PNG."""
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type "none" for this scanline
        for (r, g, b) in row:
            raw.extend((r, g, b))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


# ---------------------------------------------------------------------------
# A 5x7 pixel font — just enough glyphs to render a legible receipt.
# Each glyph is 7 strings of 5 chars; '#' is ink, ' ' is background.
# ---------------------------------------------------------------------------
_FONT = {
    "A": ["  #  ", " # # ", "#   #", "#####", "#   #", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#### ", "#   #", "#   #", "#   #", "#### "],
    "C": [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#### ", "#    ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#### ", "#    ", "#    ", "#    ", "#    "],
    "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "],
    "H": ["#   #", "#   #", "#####", "#   #", "#   #", "#   #", "#   #"],
    "I": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
    "K": ["#   #", "#  # ", "###  ", "#  # ", "#   #", "#   #", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "P": ["#### ", "#   #", "#### ", "#    ", "#    ", "#    ", "#    "],
    "R": ["#### ", "#   #", "#### ", "# #  ", "#  # ", "#   #", "#   #"],
    "S": [" ####", "#    ", " ### ", "    #", "    #", "    #", "#### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
    "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"],
    "3": ["#### ", "    #", "  ## ", "    #", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": [" ### ", "#    ", "#### ", "#   #", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", " ### ", "#   #", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
    ".": ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "],
    "$": ["  #  ", " ####", "# #  ", " ### ", "  # #", "#### ", "  #  "],
    "-": ["     ", "     ", "     ", "#####", "     ", "     ", "     "],
    ":": ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "],
    "/": ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "],
    " ": ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
}

_INK = (30, 30, 30)
_BG = (245, 245, 240)
_SCALE = 2  # each font pixel becomes 2x2 image pixels


def _blank(width: int, height: int, color: tuple[int, int, int] = _BG) -> list[list[tuple[int, int, int]]]:
    return [[color for _ in range(width)] for _ in range(height)]


def _draw_text(canvas, x: int, y: int, text: str, color=_INK) -> None:
    """Stamp `text` onto `canvas` at top-left (x, y), in the 5x7 font."""
    cx = x
    for ch in text.upper():
        glyph = _FONT.get(ch, _FONT[" "])
        for gy, line in enumerate(glyph):
            for gx, cell in enumerate(line):
                if cell != "#":
                    continue
                for sy in range(_SCALE):
                    for sx in range(_SCALE):
                        py = y + gy * _SCALE + sy
                        px = cx + gx * _SCALE + sx
                        if 0 <= py < len(canvas) and 0 <= px < len(canvas[0]):
                            canvas[py][px] = color
        cx += (5 + 1) * _SCALE  # advance one glyph + a space column
    return None


def make_receipt(path: str) -> None:
    """A small store receipt — the §3 'extract structured JSON' demo input."""
    lines = [
        "CORNER CAFE",
        "123 MAPLE ST",
        "------------",
        "LATTE      4.50",
        "MUFFIN     3.25",
        "TEA        2.75",
        "------------",
        "SUBTOTAL  10.50",
        "TAX        0.84",
        "TOTAL     11.34",
        "------------",
        "CARD  4242",
        "2026-06-26",
        "THANK YOU",
    ]
    line_h = (7 + 2) * _SCALE
    longest = max(len(line) for line in lines)
    width = 6 + longest * (5 + 1) * _SCALE + 6
    height = line_h * len(lines) + 8
    canvas = _blank(width, height)
    for i, line in enumerate(lines):
        _draw_text(canvas, 6, 4 + i * line_h, line)
    write_png(path, canvas)


def make_chart(path: str) -> None:
    """A tiny 3-bar chart — the §4 multi-image comparison demo input."""
    width, height = 130, 90
    canvas = _blank(width, height)
    # axes
    for y in range(10, 75):
        canvas[y][20] = _INK
    for x in range(20, 120):
        canvas[74][x] = _INK
    bars = [(30, 55, (200, 80, 70)), (55, 35, (80, 140, 200)), (80, 60, (90, 180, 110))]
    for bx, bh, color in bars:
        for y in range(74 - bh, 74):
            for x in range(bx, bx + 18):
                canvas[y][x] = color
    _draw_text(canvas, 24, 78, "1  2  3", _INK)
    write_png(path, canvas)


def _pdf_escape(text: str) -> str:
    """Escape the three characters a PDF text string treats specially."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_invoice_pdf(path: str) -> None:
    """A one-page invoice PDF, written by hand with the standard library only.

    A PDF is just text-and-bytes: a header, a set of numbered objects (a catalog,
    a page tree, a page, a content stream, a font), then a cross-reference table
    listing each object's byte offset, then a trailer. We build the objects, track
    where each one starts, and emit a correct xref so real parsers (and the vision
    models) accept it. This is the §11 native-PDF demo input — a *document*, not a
    screenshot of one."""
    lines = [
        "ACME CLOUD, INC.",
        "Invoice INV-2026-0042",
        "Date: 2026-06-20",
        "Bill to: Dana Rivera",
        "--------------------------------",
        "Description          Qty   Amount",
        "Pro plan (annual)      1  $240.00",
        "Extra seats            3  $108.00",
        "Priority support       1   $60.00",
        "--------------------------------",
        "Subtotal                  $408.00",
        "Tax (8%)                   $32.64",
        "Total                     $440.64",
        "Due: 2026-07-20",
    ]
    # Build the content stream: start a text object, set font + leading, position
    # the cursor, then draw each line with `Tj` and drop down with `T*`.
    body = ["BT", "/F1 12 Tf", "14 TL", "72 740 Td"]
    for line in lines:
        body.append(f"({_pdf_escape(line)}) Tj T*")
    body.append("ET")
    content = "\n".join(body).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"

    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref_pos

    with open(path, "wb") as f:
        f.write(bytes(out))


def make_note_wav(path: str, seconds: float = 1.0, freq: float = 440.0) -> None:
    """One second of a 440 Hz sine tone — a stand-in 'voice note' for §5/§6.

    It's not speech (we won't ship a recording of a real person), but it IS a
    real, valid WAV file you can hand to a speech-to-text API to see the request
    shape and the round-trip work end to end.
    """
    rate = 8000  # 8 kHz mono keeps the file a few KB
    frames = bytearray()
    for n in range(int(rate * seconds)):
        sample = int(32767 * 0.3 * math.sin(2 * math.pi * freq * n / rate))
        frames += struct.pack("<h", sample)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def main() -> None:
    make_receipt(os.path.join(HERE, "receipt.png"))
    make_chart(os.path.join(HERE, "chart.png"))
    make_note_wav(os.path.join(HERE, "note.wav"))
    make_invoice_pdf(os.path.join(HERE, "invoice.pdf"))
    for name in ("receipt.png", "chart.png", "note.wav", "invoice.pdf"):
        p = os.path.join(HERE, name)
        print(f"  wrote {name}  ({os.path.getsize(p):,} bytes)")
    print("\nAll assets generated with the standard library — no Pillow, no downloads.")


if __name__ == "__main__":
    main()
