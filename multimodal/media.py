"""
multimodal/media.py: tiny, dependency-free media helpers.

Loading an image or audio file is not the interesting part of multimodal work,
but every example needs to do it, so the boring bits live here once. There are NO
third-party dependencies. We read raw bytes and parse PNG/WAV headers by hand,
so these helpers work even before you `pip install` anything.

  load_bytes(path)        -> raw bytes + a guessed media type
  png_size(data)          -> (width, height) read straight from the PNG header
  save_bytes(path, data)  -> write bytes (for image-gen / TTS output)
"""

import os
import struct

# Map a file extension to the MIME type the providers expect.
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".pdf": "application/pdf",
}


def media_type_for(path: str) -> str:
    """Guess the MIME type from a file's extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MEDIA_TYPES:
        raise ValueError(f"Unknown media extension {ext!r} for {path!r}.")
    return _MEDIA_TYPES[ext]


def load_bytes(path: str) -> tuple[bytes, str]:
    """Read a media file; return (raw_bytes, media_type)."""
    with open(path, "rb") as f:
        return f.read(), media_type_for(path)


def save_bytes(path: str, data: bytes) -> None:
    """Write bytes to a file (e.g. a generated image or a TTS clip)."""
    with open(path, "wb") as f:
        f.write(data)


def png_size(data: bytes) -> tuple[int, int]:
    """Read (width, height) from PNG bytes without any image library.

    A PNG starts with an 8-byte signature, then the IHDR chunk whose first 8
    bytes of data are width and height as big-endian uint32s. We just unpack
    them, proof that an image's dimensions are right there in the first 24 bytes.
    """
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file.")
    # bytes 16-24 are width, height (after the 8-byte sig + 4 len + 4 'IHDR')
    width, height = struct.unpack(">II", data[16:24])
    return width, height
