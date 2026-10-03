"""
Pin the OpenAI image estimator to bills measured on the live API.

Each row is a real request to gpt-6-luna on 2026-10-03: `usage.prompt_tokens`
with the image, minus the same request without it. If OpenAI changes how it
bills images, these numbers go stale and the estimator should be re-measured,
not adjusted until the test passes.

Run it:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal.tokens import openai_image_tokens  # noqa: E402

MEASURED = [
    # width, height, detail, billed tokens
    (33, 33, "high", 4),
    (100, 100, "high", 19),
    (512, 512, "high", 307),
    (1000, 700, "high", 844),
    (1024, 1024, "high", 1228),
    (1300, 1300, "high", 2017),
    (1600, 1600, "high", 3000),
    (2048, 1024, "high", 2457),
    (2048, 2048, "high", 3000),
    (3000, 3000, "high", 3000),
    (4096, 2048, "high", 2940),
    (4096, 4096, "high", 3000),
    (6000, 1000, "high", 2880),
    # The sizes examples/09 prints. The phone and 4K rows are the ones that
    # caught the shrink step rounding the wrong way.
    (130, 90, "high", 18),
    (192, 260, "high", 64),
    (585, 1266, "high", 912),
    (1170, 2532, "high", 2851),
    (3840, 2160, "high", 2930),
    (100, 100, "low", 19),
    (512, 512, "low", 307),
    (3000, 3000, "low", 307),
    (1300, 1300, "auto", 2017),
    (3000, 3000, "auto", 10603),
    (1170, 2532, "auto", 3552),
    (3840, 2160, "auto", 9792),
]


class TestOpenAIImageTokens(unittest.TestCase):
    def test_matches_measured_bills(self):
        for w, h, detail, billed in MEASURED:
            with self.subTest(size=f"{w}x{h}", detail=detail):
                self.assertEqual(openai_image_tokens(w, h, detail).tokens, billed)

    def test_auto_rejects_what_the_api_rejects(self):
        # 6000x6000 at "auto" came back as a 400: 35,344 patches, limit 30,000.
        with self.assertRaises(ValueError):
            openai_image_tokens(6000, 6000, "auto")


if __name__ == "__main__":
    unittest.main()
