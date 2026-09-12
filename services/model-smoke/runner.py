"""Aggregate readiness check for the six independent model services."""

from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

SERVICES = (
    "parakeet",
    "diarization",
    "resnet293",
    "eres2net",
    "wavlm",
    "text_embeddings",
)


def main() -> int:
    timeout = float(os.environ.get("SMOKE_TIMEOUT_SECONDS", "180"))
    deadline = time.monotonic() + timeout
    results: dict[str, object] = {}
    while time.monotonic() < deadline:
        pending = []
        for service in SERVICES:
            try:
                with urlopen(f"http://{service}:8080/ready", timeout=3) as response:
                    results[service] = json.load(response)
            except (OSError, URLError, TimeoutError) as exc:
                pending.append((service, str(exc)))
        if not pending:
            print(json.dumps({"status": "passed", "services": results}, indent=2, sort_keys=True))
            return 0
        time.sleep(2)
    print(json.dumps({"status": "failed", "pending": pending, "services": results}, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
