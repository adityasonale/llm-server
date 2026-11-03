"""
Quick manual test client for the running server.

Usage (server must be up on :8000):
    python script.py                              # default: minichat, local
    python script.py --prompt "Explain gravity in one line."

Reads the API key from .env (API_KEYS) so it is never hardcoded here.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()


def _api_key() -> str:
    keys = json.loads(os.getenv("API_KEYS", "[]"))
    if not keys:
        raise SystemExit("No API_KEYS found in .env")
    return keys[0]["api_key"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/generate")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default="minichat-1.5-3b")
    parser.add_argument("--system", default="Answer in one short sentence.")
    parser.add_argument("--prompt", default="What is 2+2?")
    parser.add_argument("--timeout", type=int, default=600)  # first call downloads weights
    args = parser.parse_args()

    body = json.dumps({
        "system_prompt": args.system,
        "user_prompt": args.prompt,
        "model": args.model,
        "provider": args.provider,
    }).encode()

    req = urllib.request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": _api_key()},
    )

    print(f"-> {args.provider}/{args.model}: {args.prompt!r}")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            elapsed = time.perf_counter() - started
            data = json.loads(resp.read())
            print(f"HTTP {resp.status} in {elapsed:.1f}s")
            print("content:", data.get("content"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"failed after {time.perf_counter() - started:.1f}s: {e}")


if __name__ == "__main__":
    main()
