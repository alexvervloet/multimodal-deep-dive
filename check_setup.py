"""
Setup check — run this first.
=============================

    secrun python check_setup.py

Checks your Python version, the installed packages, your chosen PROVIDER, the API
key that provider needs, and that the sample assets are present — and tells you
exactly what to fix. It also prints which multimodal capabilities your provider
supports, because they differ (vision works everywhere; audio and image
generation are OpenAI-only).

Makes NO API calls. Uses only the standard library, so it runs even before
`pip install`.
"""

import importlib.util
import os
import sys

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def ok(msg):
    print(f"  {_c('✓', '32')} {msg}")


def warn(msg):
    print(f"  {_c('!', '33')} {msg}")


def fail(msg):
    print(f"  {_c('✗', '31')} {msg}")


HERE = os.path.dirname(os.path.abspath(__file__))


def _read_env_file():
    env_path = os.path.join(HERE, ".env")
    values = {}
    if not os.path.exists(env_path):
        return None
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _get(env, name):
    return os.getenv(name) or (env or {}).get(name, "")


ALWAYS = [
    ("dotenv", "python-dotenv", "loads PROVIDER/config from .env"),
    ("rich", "rich", "tables and output in the examples and capstone"),
]
PROVIDER_DEPS = {
    "openai": [("openai", "openai", "OpenAI vision, Whisper, TTS, image gen")],
    "claude": [("anthropic", "anthropic", "Claude vision (messages)")],
}
PROVIDER_KEYS = {
    "openai": [("OPENAI_API_KEY", "sk-", "sk-your-openai-key-here")],
    "claude": [("ANTHROPIC_API_KEY", "sk-ant-", "sk-ant-your-key-here")],
}
# Capabilities per provider — kept in sync with multimodal/providers.py so the
# check works even before the SDKs are installed.
CAPABILITIES = {
    "openai": {"vision": True, "audio (STT/TTS)": True, "image generation": True},
    "claude": {"vision": True, "audio (STT/TTS)": False, "image generation": False},
}
ASSETS = ["receipt.png", "chart.png", "note.wav", "invoice.pdf"]


def check_python():
    print("Python version")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        ok(f"Python {major}.{minor} (3.10+ required)")
        return True
    fail(f"Python {major}.{minor} — this repo needs Python 3.10 or newer.")
    print("    Install a newer Python from https://www.python.org/downloads/")
    return False


def check_provider(env):
    print("\nProvider")
    provider = (_get(env, "PROVIDER") or "openai").strip().lower()
    if provider in PROVIDER_DEPS:
        ok(f"PROVIDER = {provider}")
        return provider
    fail(f"PROVIDER = {provider!r} is not recognized.")
    print("    Set PROVIDER=openai or PROVIDER=claude in .env.")
    return None


def check_dependencies(provider):
    print("\nDependencies")
    needed = ALWAYS + PROVIDER_DEPS.get(provider, [])
    missing = []
    for import_name, pip_name, purpose in needed:
        if importlib.util.find_spec(import_name) is not None:
            ok(f"{pip_name} — {purpose}")
        else:
            fail(f"{pip_name} MISSING — {purpose}")
            missing.append(pip_name)
    if missing:
        print("\n    Install everything with:")
        print("        pip install -r requirements.txt")
    return not missing


def check_keys(env, provider):
    print("\nAPI key")
    if env is None:
        fail(".env file not found.")
        print("    Create it with:  cp .env.example .env")
        return False
    all_ok = True
    for name, prefix, placeholder in PROVIDER_KEYS.get(provider, []):
        value = _get(env, name)
        if not value or value == placeholder:
            fail(f"{name} is not set.")
            print("    Store it in your OS keychain and run `secrun python check_setup.py` — see SECRETS.md.")
            all_ok = False
        elif not value.startswith(prefix):
            warn(f"{name} is set but doesn't start with '{prefix}'. Double-check it.")
        else:
            ok(f"{name} is set and looks right.")
    return all_ok


def check_assets():
    print("\nSample assets")
    assets_dir = os.path.join(HERE, "assets")
    all_ok = True
    for name in ASSETS:
        path = os.path.join(assets_dir, name)
        if os.path.exists(path):
            ok(f"{name} ({os.path.getsize(path):,} bytes)")
        else:
            fail(f"{name} MISSING")
            all_ok = False
    if not all_ok:
        print("\n    Regenerate them (offline, no key) with:")
        print("        python assets/make_assets.py")
    return all_ok


def show_capabilities(provider):
    print("\nCapabilities for this provider")
    for cap, available in CAPABILITIES.get(provider, {}).items():
        if available:
            ok(f"{cap}")
        else:
            warn(f"{cap} — not supported on {provider}; those examples will say so and skip")


def main():
    print(_c("Checking your setup for the Multimodal deep dive...\n", "1"))
    env = _read_env_file()
    py = check_python()
    provider = check_provider(env)
    if provider is None:
        print(_c("\nFix PROVIDER in .env, then run this again.", "1;31"))
        return 1
    deps = check_dependencies(provider)
    keys = check_keys(env, provider)
    assets = check_assets()
    show_capabilities(provider)

    print()
    if py and deps and keys and assets:
        print(_c("All set! 🎉", "1;32"))
        print("Start here:  python examples/01_vision_offline.py")
        print("(Examples 01 and 09 are offline and need no key.)")
        return 0
    print(_c("Not ready yet — fix the ✗ items above, then run this again.", "1;31"))
    print("(Examples 01 and 09 are offline, so you can run those without a key.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
