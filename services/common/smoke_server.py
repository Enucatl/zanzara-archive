"""Small internal-only readiness server shared by model smoke services."""

from __future__ import annotations

import json
import os
import platform
import socketserver
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any

import torch
import torchaudio
import transformers


def run_service(model_id: str, infer: Callable[[Path], dict[str, Any]]) -> None:
    """Run one real model inference, persist aggregate evidence, then serve readiness."""

    audio_path = Path(os.environ.get("SMOKE_AUDIO", "/smoke/audio.wav"))
    evidence_path = Path(os.environ.get("EVIDENCE_PATH", f"/evidence/{model_id}.json"))
    if not audio_path.is_file():
        raise FileNotFoundError(f"smoke audio is missing: {audio_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("the model smoke requires a CUDA device")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = infer(audio_path)
    torch.cuda.synchronize()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    payload = {
        "model": model_id,
        "status": "passed",
        "shape": result.pop("shape"),
        "finite_values": result.pop("finite_values", True),
        "elapsed_ms": elapsed_ms,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
        "gpu": torch.cuda.get_device_name(0),
        "driver_cuda": torch.version.cuda,
        "model_source": os.environ.get("MODEL_SOURCE", model_id),
        "model_revision": os.environ.get("MODEL_REVISION", "unknown"),
        "image_ref": os.environ.get("IMAGE_REF", "unknown"),
        "base_image_ref": os.environ.get("BASE_IMAGE_REF", "unknown"),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": transformers.__version__,
        },
        "audio": {
            "path": "public ModelScope ERes2Net example",
            "provenance": "real public example audio; no archive content",
        },
        **result,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _serve(payload)


class _Handler(BaseHTTPRequestHandler):
    payload: dict[str, Any] = {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/health", "/ready", "/v1/smoke"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = json.dumps(self.payload, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def _serve(payload: dict[str, Any]) -> None:
    _Handler.payload = payload
    server = socketserver.TCPServer(("0.0.0.0", 8080), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    while True:
        time.sleep(3600)
