"""Small internal-only readiness server shared by model smoke services."""

from __future__ import annotations

import json
import os
import platform
import socketserver
import time
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import torch
import torchaudio
import transformers

PostInference = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_service(
    model_id: str,
    infer: Callable[[Path], dict[str, Any]],
    post_infer: PostInference | None = None,
) -> None:
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
    _serve(payload, post_infer)


class _Handler(BaseHTTPRequestHandler):
    payload: dict[str, Any] = {}
    post_infer: PostInference | None = None

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

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/audio/transcriptions" or self.post_infer is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            content_length = -1
        if content_length < 0 or content_length > 40_000_000:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            raw = self.rfile.read(content_length)
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, Mapping):
                raise ValueError("request must be a JSON object")
            request_id = request.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                request_id = f"parakeet-{uuid4().hex}"
            response = {
                "request_id": request_id,
                "status": "ok",
                "data": dict(type(self).post_infer(request)),
            }
            status = HTTPStatus.OK
        except Exception as exc:  # noqa: BLE001 - internal endpoint returns typed failure JSON
            request_id = (
                request.get("request_id")
                if "request" in locals() and isinstance(request, Mapping)
                else f"parakeet-{uuid4().hex}"
            )
            if not isinstance(request_id, str) or not request_id:
                request_id = f"parakeet-{uuid4().hex}"
            response = {
                "request_id": request_id,
                "status": "error",
                "error": {
                    "code": "inference_error",
                    "message": str(exc),
                    "retryable": False,
                    "request_id": request_id,
                },
            }
            status = HTTPStatus.BAD_REQUEST
        body = json.dumps(response, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def _serve(payload: dict[str, Any], post_infer: PostInference | None = None) -> None:
    _Handler.payload = payload
    _Handler.post_infer = post_infer
    server = socketserver.ThreadingTCPServer(("0.0.0.0", 8080), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    while True:
        time.sleep(3600)
