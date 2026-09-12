# HTTP client policy and inventory

This document records the P1-10 dependency audit. The inventory covers the
root application, all six model-service packages, tests, Docker/Compose
metadata, and every committed `uv.lock` file.

## Policy

- New application-managed HTTP calls use `niquests` when it supports the
  required timeout, TLS, proxy, streaming, retry, response-validation, and
  request-accounting behavior.
- Direct `requests` or `httpx` dependencies/imports anywhere require a written
  exception here. A transitive occurrence is tracked separately and is not an
  application client choice.
- Standard-library HTTP code is retained when it is the smallest stable
  implementation. It is not replaced only to reduce a package count.

## Direct and source inventory

| Scope | Direct client dependency or import | Classification | Decision |
| --- | --- | --- | --- |
| Root `pyproject.toml` | `niquests>=3.21,<4` in production; `httpx2>=2,<3` in `dev` | `niquests`: application-managed inference client; `httpx2`: test-only framework transport | `niquests` is used by `src/zanzara_archive/inference.py`; `httpx2` is retained because current Starlette imports it for `TestClient` (with a deprecated `httpx` fallback). |
| Root production dependencies | `niquests` only | Application HTTP client | Migrated the OpenAI-compatible inference POST adapter from `urllib.request` to `niquests`. |
| `src/zanzara_archive/inference.py` | `niquests.post` | Application-managed provider/service client | Uses explicit timeout, TLS verification, proxy-compatible defaults, redirects, streaming, zero retries, bounded response reads, and typed timeout mapping. |
| Other `src/` and `services/` Python code | No `httpx`, `requests`, or `niquests` imports/calls | No application-managed client | No migration needed. |
| `tests/` | `fastapi.testclient.TestClient`; no direct client import | Framework test transport | Covered by the root dev exception above. |
| Dockerfiles and Compose | No third-party HTTP client installation; Compose healthcheck uses standard-library `urllib.request` with a bounded timeout and JSON validation | Standard-library operational probe | Retained under the standard-library policy. |

No direct production `requests` or `httpx` use remains. The direct production
HTTP package is `niquests`, used by the inference adapter; the only direct test
HTTP package is framework-preferred `httpx2`. `niquests` is not used for the
standard-library health probes because those probes do not form an
application-managed provider/service client.

## Transitive lockfile inventory

The paths below are the shortest application-root paths reported by `uv tree
--invert`. Repeated appearances through another top-level dependency resolve
to the same locked package and do not represent duplicate installations.

| Environment | `niquests` path | `httpx` path | `requests` path |
| --- | --- | --- | --- |
| Root | `zanzara-archive -> niquests` | No `httpx` package; `zanzara-archive (dev) -> httpx2` is the framework test transport | None |
| `services/parakeet` | `parakeet-service -> accelerate -> huggingface-hub -> httpx` (also via `transformers`) | `parakeet-service -> librosa -> pooch -> requests` |
| `services/diarization` | `diarization-service -> pyannote-audio -> huggingface-hub -> httpx` (also via `transformers`) | `diarization-service -> pyannote-audio -> opentelemetry-exporter-otlp -> opentelemetry-exporter-otlp-proto-http -> requests`; also `pyannoteai-sdk -> requests` |
| `services/resnet293` | `resnet293-service -> transformers -> huggingface-hub -> httpx` | `resnet293-service -> wespeaker -> s3prl -> requests`; also `openai-whisper -> tiktoken -> requests` |
| `services/eres2net` | `eres2net-service -> transformers -> huggingface-hub -> httpx` | None |
| `services/wavlm` | `wavlm-service -> transformers -> huggingface-hub -> httpx` | None |
| `services/text-embeddings` | `text-embeddings-service -> accelerate -> huggingface-hub -> httpx` (also via `transformers`) | None |

The service occurrences are owned by locked model/download/telemetry
dependencies, not by Zanzara application code. Replacing them would require
changing supported upstream package contracts and is outside this issue.

## Lock, image, and semantic impact

The root lock was re-generated for the `niquests` inference adapter and the
framework-supported `httpx2` test transport. The six service locks remain
unchanged and contain one locked `httpx` version (`0.28.1`) where the paths
above require it. The locked `requests` version is `2.34.2` where required by
the service graphs. No duplicate `httpx`/`requests` installation was
introduced.

The compressed wheel sizes observed in the locked root graph are:

- The old root test transport closure removed `certifi` 2026.7.22 (136,983
  bytes), `httpx` 0.28.1 (73,517 bytes), and `httpcore` 1.0.9 (78,784 bytes).
- The replacement test transport closure adds `httpx2` 2.12.0 (95,427 bytes),
  `httpcore2` 2.12.0 (83,074 bytes), and `truststore` 0.10.4 (18,660 bytes),
  for a decrease of 92,123 compressed bytes. The `httpx2-jsfetch` lock entry
  is emscripten-only and is not installed on the target platform.
- The niquests closure adds `charset-normalizer` 3.5.1 (251,240 bytes),
  `jh2` 5.0.14 (396,710 bytes), `niquests` 3.21.1 (230,450 bytes), `qh3`
  2.0.3 (2,509,787 bytes), `urllib3-future` 2.24.908 (820,384 bytes), and
  `wassima` 2.1.4 (134,348 bytes), totaling 4,342,919 compressed bytes.
- The complete locked root graph therefore grows by 4,250,796 compressed
  bytes. Exact locked wheel hashes and all platform-specific entries are in
  the private acceptance evidence.
- The service `httpx` and `requests` wheels remain 73,517 and 73,075 bytes
  respectively.

The `httpx2` change follows Starlette's supported TestClient transport and
removes the deprecated fallback. The niquests closure is a required production
dependency for the migrated inference adapter; its measured dependency impact
is recorded in the private evidence. Docker metadata did not need a
package-install change because the application image consumes the locked root
environment.

The migrated inference client preserves timeout, TLS verification, proxy
environment behavior, redirect handling, streaming, zero implicit retries,
bounded response reads, and error-body/typed-timeout behavior. Existing
response validation remains in the adapter parser and request accounting
remains outside the transport. The framework exception preserves the existing
ASGI `TestClient` transport and its test coverage without the deprecated
`httpx` fallback. The standard-library Compose probe remains bounded and
unchanged.

## Reproducible checks

Run from the repository root:

```bash
rg -n "httpx|requests|niquests" pyproject.toml services src tests docs
uv tree
for service in services/parakeet services/diarization services/resnet293 \
  services/eres2net services/wavlm services/text-embeddings; do
  uv tree --directory "$service"
done
uv run pytest tests/test_parakeet_adapter.py tests/test_diarization_adapter.py -q
uv lock --check
for service in services/parakeet services/diarization services/resnet293 \
  services/eres2net services/wavlm services/text-embeddings; do
  uv lock --check --directory "$service"
done
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The complete CPU test suite is the relevant semantic check. No credentials,
network inference, paid request, or private network trace is needed or
included in this inventory.
