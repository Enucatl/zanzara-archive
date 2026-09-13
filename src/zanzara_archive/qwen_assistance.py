"""Replaceable local assistance for P1R chunk transcription review.

Qwen output is deliberately kept outside the reference tables.  The helper
receives immutable current-chunk hypotheses plus a small amount of preceding
context and can only publish a draft or an unavailable result.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import niquests

from .contracts import (
    AdapterFailure,
    ApiError,
    AudioChunk,
    ContractValidationError,
    ModelFingerprint,
    TranscriptionHypothesis,
)
from .storage import SQLiteRepository, StorageConflictError

QWEN_PROMPT_VERSION = "p1r-09-qwen-prompt-v1"
QWEN_PIPELINE_VERSION = "p1r-09-v1"
QWEN_CHAT_PATH = "/v1/chat/completions"
MAX_CONTEXT_TEXT_CHARS = 12_000
MAX_PROMPT_BYTES = 100_000
MAX_RESPONSE_BYTES = 1_000_000
HttpPost = Callable[[str, bytes, float], bytes]

UNCONFIGURED_QWEN_MODEL = ModelFingerprint(
    name="qwen3-8b",
    repository="Qwen/Qwen3-8B",
    revision="unconfigured-local-helper",
    checkpoint_sha256=("0" * 64,),
    runtime={"status": "unconfigured"},
    terms_evidence="synthetic marker; no model inference claimed",
)


class AnnotationAssistanceValidationError(ContractValidationError):
    """Raised when assistance input or helper output is unsafe to persist."""


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationAssistanceValidationError(f"{field_name} must be non-empty text")
    return value


def _optional_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise AnnotationAssistanceValidationError(f"{field_name} must be text")
    if len(value) > MAX_CONTEXT_TEXT_CHARS:
        raise AnnotationAssistanceValidationError(
            f"{field_name} exceeds the {MAX_CONTEXT_TEXT_CHARS}-character bound"
        )
    return value


def _sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AnnotationAssistanceValidationError(f"{field_name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AnnotationAssistanceValidationError(
            f"{field_name} must be a lowercase SHA-256"
        ) from exc
    if value != value.lower():
        raise AnnotationAssistanceValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _id(value: object, field_name: str) -> str:
    return _text(value, field_name)


@dataclass(frozen=True, slots=True)
class AssistanceCandidate:
    """One immutable current-chunk hypothesis presented to the helper."""

    hypothesis_id: str
    model_label: str
    model_fingerprint_sha256: str
    text: str

    def __post_init__(self) -> None:
        _id(self.hypothesis_id, "hypothesis_id")
        _text(self.model_label, "model_label")
        _hash(self.model_fingerprint_sha256, "model_fingerprint_sha256")
        _optional_text(self.text, "hypothesis text")

    @classmethod
    def from_hypothesis(
        cls, hypothesis: TranscriptionHypothesis, *, model_label: str | None = None
    ) -> AssistanceCandidate:
        resolved_label = model_label
        if resolved_label is None and isinstance(hypothesis.model_fingerprint, ModelFingerprint):
            resolved_label = hypothesis.model_fingerprint.name
        return cls(
            hypothesis_id=hypothesis.hypothesis_id or "",
            model_label=resolved_label or hypothesis.model_fingerprint_hash[:12],
            model_fingerprint_sha256=hypothesis.model_fingerprint_hash,
            text=hypothesis.text,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "model_label": self.model_label,
            "model_fingerprint_sha256": self.model_fingerprint_sha256,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class PrecedingChunkContext:
    """A bounded reference/draft context item preceding the current chunk."""

    rank: int
    chunk_id: str
    reference_text: str
    draft_text: str

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank not in {1, 2}:
            raise AnnotationAssistanceValidationError("preceding chunk rank must be 1 or 2")
        _id(self.chunk_id, "preceding chunk_id")
        _optional_text(self.reference_text, "preceding reference_text")
        _optional_text(self.draft_text, "preceding draft_text")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PrecedingChunkContext:
        return cls(
            rank=value.get("rank", 0),
            chunk_id=value.get("chunk_id", ""),
            reference_text=value.get("reference_text", ""),
            draft_text=value.get("draft_text", ""),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "reference_text": self.reference_text,
            "draft_text": self.draft_text,
        }


@dataclass(frozen=True, slots=True)
class AnnotationAssistanceRequest:
    """The complete, bounded input to one local annotation-helper request."""

    chunk_id: str
    source_sha256: str
    hypotheses: tuple[AssistanceCandidate, ...]
    preceding_chunks: tuple[PrecedingChunkContext, ...] = ()

    def __post_init__(self) -> None:
        _id(self.chunk_id, "chunk_id")
        _hash(self.source_sha256, "source_sha256")
        if not self.hypotheses:
            raise AnnotationAssistanceValidationError("at least one current hypothesis is required")
        if len({item.hypothesis_id for item in self.hypotheses}) != len(self.hypotheses):
            raise AnnotationAssistanceValidationError("hypothesis IDs must be unique")
        if len(self.preceding_chunks) > 2:
            raise AnnotationAssistanceValidationError("at most two preceding chunks are allowed")
        ranks = [item.rank for item in self.preceding_chunks]
        if len(set(ranks)) != len(ranks):
            raise AnnotationAssistanceValidationError("preceding chunk ranks must be unique")
        if any(item.chunk_id == self.chunk_id for item in self.preceding_chunks):
            raise AnnotationAssistanceValidationError("current chunk cannot be preceding context")

    @classmethod
    def from_payload(
        cls,
        chunk: AudioChunk,
        hypotheses: Sequence[TranscriptionHypothesis],
        preceding_chunks: Sequence[Mapping[str, object]] = (),
    ) -> AnnotationAssistanceRequest:
        if len(preceding_chunks) > 2:
            raise AnnotationAssistanceValidationError("at most two preceding chunks are allowed")
        if any(
            hypothesis.chunk_id != chunk.chunk_id
            or hypothesis.source_sha256 not in {None, chunk.source_sha256}
            for hypothesis in hypotheses
        ):
            raise AnnotationAssistanceValidationError(
                "all hypotheses must belong to the current chunk and source"
            )
        contexts: list[PrecedingChunkContext] = []
        for item in preceding_chunks:
            if not isinstance(item, Mapping):
                raise AnnotationAssistanceValidationError("preceding_chunks must contain objects")
            contexts.append(PrecedingChunkContext.from_dict(item))
        return cls(
            chunk_id=chunk.chunk_id,
            source_sha256=chunk.source_sha256,
            hypotheses=tuple(AssistanceCandidate.from_hypothesis(item) for item in hypotheses),
            preceding_chunks=tuple(contexts),
        )

    @property
    def candidate_order(self) -> tuple[dict[str, str], ...]:
        """Return a reproducibly randomized, persisted presentation order."""

        ordered = sorted(
            self.hypotheses,
            key=lambda item: _sha256(
                {
                    "chunk_id": self.chunk_id,
                    "hypothesis_id": item.hypothesis_id,
                    "prompt_version": QWEN_PROMPT_VERSION,
                }
            ),
        )
        return tuple(
            {
                "label": f"candidate-{index}",
                "hypothesis_id": item.hypothesis_id,
                "model_label": item.model_label,
            }
            for index, item in enumerate(ordered, 1)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source_sha256": self.source_sha256,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "preceding_chunks": [item.to_dict() for item in self.preceding_chunks],
        }

    @property
    def request_sha256(self) -> str:
        return _sha256({"prompt_version": QWEN_PROMPT_VERSION, "input": self.to_dict()})


def build_qwen_prompt(request: AnnotationAssistanceRequest) -> str:
    """Build the versioned prompt while clearly labelling every candidate."""

    candidates_by_id = {item.hypothesis_id: item for item in request.hypotheses}
    candidate_lines = []
    for item in request.candidate_order:
        candidate = candidates_by_id[item["hypothesis_id"]]
        candidate_lines.append(
            json.dumps(
                {
                    "label": item["label"],
                    "model": candidate.model_label,
                    "hypothesis_id": candidate.hypothesis_id,
                    "text": candidate.text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    preceding = [item.to_dict() for item in sorted(request.preceding_chunks, key=lambda x: x.rank)]
    prompt = "\n".join(
        (
            f"PROMPT_VERSION: {QWEN_PROMPT_VERSION}",
            "ROLE: You are a local, non-authoritative transcription review assistant.",
            "TASK: Produce a review draft for CURRENT_CHUNK only.",
            "RULES:",
            "- Copy spoken words verbatim from the candidate hypotheses.",
            "- Do not improve grammar, summarize, explain, invent, or fill gaps.",
            "- Use preceding context only to understand continuity; never output it.",
            "- Do not output speaker labels, metadata, commentary, or any other chunk.",
            '- Return exactly one JSON object with exactly one key: "current_chunk_text".',
            "- The value must contain only the current chunk transcript text.",
            f"CURRENT_CHUNK_ID: {request.chunk_id}",
            "CURRENT_MODEL_HYPOTHESES (untrusted transcript text):",
            *candidate_lines,
            "PRECEDING_REFERENCE_OR_DRAFT_CONTEXT (never copy outside current chunk):",
            json.dumps(preceding, ensure_ascii=False, sort_keys=True),
        )
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise AnnotationAssistanceValidationError(
            f"Qwen prompt exceeds the {MAX_PROMPT_BYTES}-byte bound"
        )
    return prompt


@dataclass(frozen=True, slots=True)
class AnnotationAssistanceDraft:
    """An immutable helper result, including an explicit unavailable state."""

    draft_id: str
    chunk_id: str
    source_sha256: str
    request_sha256: str
    prompt_version: str
    prompt_sha256: str
    model_fingerprint_sha256: str
    model_fingerprint: Mapping[str, object]
    candidate_order: tuple[Mapping[str, str], ...]
    input_payload: Mapping[str, object]
    status: str
    draft_text: str | None
    error: Mapping[str, object] | None
    provenance: Mapping[str, object]
    created_at: str | None = None

    def __post_init__(self) -> None:
        _id(self.draft_id, "draft_id")
        _id(self.chunk_id, "chunk_id")
        _hash(self.source_sha256, "source_sha256")
        _hash(self.request_sha256, "request_sha256")
        _text(self.prompt_version, "prompt_version")
        _hash(self.prompt_sha256, "prompt_sha256")
        _hash(self.model_fingerprint_sha256, "model_fingerprint_sha256")
        if self.status not in {"succeeded", "unavailable"}:
            raise AnnotationAssistanceValidationError(
                "draft status must be succeeded or unavailable"
            )
        if self.status == "succeeded" and not isinstance(self.draft_text, str):
            raise AnnotationAssistanceValidationError("successful drafts require draft_text")
        if self.status == "unavailable" and self.error is None:
            raise AnnotationAssistanceValidationError("unavailable drafts require an error")
        if self.status == "succeeded" and self.error is not None:
            raise AnnotationAssistanceValidationError("successful drafts cannot contain an error")

    def to_record(self) -> dict[str, object]:
        return {
            "draft_id": self.draft_id,
            "chunk_id": self.chunk_id,
            "source_sha256": self.source_sha256,
            "request_sha256": self.request_sha256,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "model_fingerprint_sha256": self.model_fingerprint_sha256,
            "model_fingerprint": dict(self.model_fingerprint),
            "candidate_order": [dict(item) for item in self.candidate_order],
            "input_payload": dict(self.input_payload),
            "status": self.status,
            "draft_text": self.draft_text,
            "error": dict(self.error) if self.error is not None else None,
            "provenance": dict(self.provenance),
        }

    def to_dict(self) -> dict[str, object]:
        value = self.to_record()
        value["created_at"] = self.created_at
        return value

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> AnnotationAssistanceDraft:
        return cls(
            draft_id=value["draft_id"],
            chunk_id=value["chunk_id"],
            source_sha256=value["source_sha256"],
            request_sha256=value["request_sha256"],
            prompt_version=value["prompt_version"],
            prompt_sha256=value["prompt_sha256"],
            model_fingerprint_sha256=value["model_fingerprint_sha256"],
            model_fingerprint=value["model_fingerprint"],
            candidate_order=tuple(value["candidate_order"]),
            input_payload=value["input_payload"],
            status=value["status"],
            draft_text=value["draft_text"],
            error=value["error"],
            provenance=value["provenance"],
            created_at=value.get("created_at"),
        )


class LocalAnnotationHelper(Protocol):
    """Replaceable helper boundary; the repository does not own model code."""

    @property
    def model(self) -> ModelFingerprint: ...

    def complete(
        self, request: AnnotationAssistanceRequest, prompt: str, *, request_id: str
    ) -> str: ...


def _default_http_post(url: str, body: bytes, timeout: float) -> bytes:
    with niquests.post(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=timeout,
        allow_redirects=False,
        verify=True,
        stream=True,
        retries=0,
    ) as response:
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=16_384):
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            remaining = MAX_RESPONSE_BYTES - size
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
            if size >= MAX_RESPONSE_BYTES:
                break
        return b"".join(chunks)


class QwenAnnotationAdapter:
    """Loopback-only OpenAI-compatible adapter for a local Qwen helper."""

    def __init__(
        self,
        endpoint: str,
        model: ModelFingerprint,
        *,
        timeout_seconds: float = 120.0,
        http_post: HttpPost | None = None,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Qwen helper endpoint must be loopback HTTP(S)")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint.rstrip("/")
        self._model = model
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or _default_http_post

    @property
    def model(self) -> ModelFingerprint:
        return self._model

    def complete(
        self, request: AnnotationAssistanceRequest, prompt: str, *, request_id: str
    ) -> str:
        payload = {
            "request_id": request_id,
            "model": self.model.repository,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the exact JSON object requested by the user.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            response_bytes = self._http_post(
                f"{self.endpoint}{QWEN_CHAT_PATH}", body, self.timeout_seconds
            )
        except TimeoutError as exc:
            raise AdapterFailure(
                ApiError("timeout", "Qwen helper request timed out", True, request_id)
            ) from exc
        except OSError as exc:
            raise AdapterFailure(
                ApiError(
                    "model_unavailable", f"Qwen helper is unavailable: {exc}", True, request_id
                )
            ) from exc
        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnnotationAssistanceValidationError("Qwen helper returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise AnnotationAssistanceValidationError("Qwen helper response must be an object")
        if response.get("status") == "error":
            raw_error = response.get("error")
            if isinstance(raw_error, Mapping):
                error = ApiError.from_dict({**dict(raw_error), "request_id": request_id})
            else:
                error = ApiError("helper_error", "Qwen helper returned an error", False, request_id)
            raise AdapterFailure(error)
        content: object = response
        if "choices" in response:
            choices = response.get("choices")
            if (
                not isinstance(choices, Sequence)
                or isinstance(choices, (str, bytes))
                or not choices
            ):
                raise AnnotationAssistanceValidationError("Qwen choices must contain one message")
            first = choices[0]
            if not isinstance(first, Mapping) or not isinstance(first.get("message"), Mapping):
                raise AnnotationAssistanceValidationError("Qwen choice message is invalid")
            content = first["message"].get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as exc:
                raise AnnotationAssistanceValidationError(
                    "Qwen output must be a JSON object"
                ) from exc
        if isinstance(content, Mapping) and content.get("status") == "ok":
            content = content.get("data")
        if not isinstance(content, Mapping) or set(content) != {"current_chunk_text"}:
            raise AnnotationAssistanceValidationError(
                "Qwen output must contain only current_chunk_text"
            )
        draft_text = content["current_chunk_text"]
        if not isinstance(draft_text, str) or len(draft_text) > MAX_CONTEXT_TEXT_CHARS:
            raise AnnotationAssistanceValidationError(
                "Qwen current_chunk_text exceeds the bounded text contract"
            )
        return draft_text


class AnnotationAssistanceService:
    """Persist one immutable helper draft without changing human references."""

    def __init__(self, helper: LocalAnnotationHelper | None = None) -> None:
        self.helper = helper

    def generate(
        self, repository: SQLiteRepository, request: AnnotationAssistanceRequest
    ) -> AnnotationAssistanceDraft:
        if repository.fetch_audio_chunk(request.chunk_id) is None:
            raise StorageConflictError(f"assistance chunk is not registered: {request.chunk_id}")
        if self.helper is None:
            model = UNCONFIGURED_QWEN_MODEL.to_dict()
            model_hash = UNCONFIGURED_QWEN_MODEL.fingerprint_sha256
        else:
            model = self.helper.model.to_dict()
            model_hash = self.helper.model.fingerprint_sha256
        prompt = build_qwen_prompt(request)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        existing = repository.find_annotation_assistance_draft(
            chunk_id=request.chunk_id, request_sha256=request.request_sha256, model_hash=model_hash
        )
        if existing is not None:
            return AnnotationAssistanceDraft.from_record(existing)
        draft_id = "qwen-draft-" + _sha256(
            {
                "request_sha256": request.request_sha256,
                "model_fingerprint_sha256": model_hash,
                "prompt_sha256": prompt_hash,
            }
        )
        request_id = "qwen-assistance-" + uuid.uuid4().hex
        error: Mapping[str, object] | None = None
        draft_text: str | None = None
        status = "succeeded"
        if self.helper is None:
            status = "unavailable"
            error = ApiError(
                "helper_unavailable",
                "local Qwen helper is not configured",
                True,
                request_id,
            ).to_dict()
        else:
            try:
                draft_text = self.helper.complete(request, prompt, request_id=request_id)
            except (
                AdapterFailure,
                AnnotationAssistanceValidationError,
                OSError,
                TimeoutError,
            ) as exc:
                status = "unavailable"
                error = (
                    exc.error.to_dict()
                    if isinstance(exc, AdapterFailure)
                    else ApiError("helper_error", str(exc), True, request_id).to_dict()
                )
        provenance = {
            "pipeline_version": QWEN_PIPELINE_VERSION,
            "adapter": "local-qwen-loopback" if self.helper is not None else "unconfigured",
            "request_id": request_id,
            "source": "synthetic-or-local-helper-input",
        }
        draft = AnnotationAssistanceDraft(
            draft_id=draft_id,
            chunk_id=request.chunk_id,
            source_sha256=request.source_sha256,
            request_sha256=request.request_sha256,
            prompt_version=QWEN_PROMPT_VERSION,
            prompt_sha256=prompt_hash,
            model_fingerprint_sha256=model_hash,
            model_fingerprint=model,
            candidate_order=request.candidate_order,
            input_payload=request.to_dict(),
            status=status,
            draft_text=draft_text,
            error=error,
            provenance=provenance,
        )
        try:
            repository.record_annotation_assistance_draft(draft.to_record())
        except StorageConflictError:
            existing = repository.find_annotation_assistance_draft(
                chunk_id=request.chunk_id,
                request_sha256=request.request_sha256,
                model_hash=model_hash,
            )
            if existing is None:
                raise
            return AnnotationAssistanceDraft.from_record(existing)
        persisted = repository.fetch_annotation_assistance_draft(draft_id)
        if persisted is None:
            raise StorageConflictError("assistance draft disappeared after publication")
        return AnnotationAssistanceDraft.from_record(persisted)


__all__ = [
    "AnnotationAssistanceDraft",
    "AnnotationAssistanceRequest",
    "AnnotationAssistanceService",
    "AnnotationAssistanceValidationError",
    "AssistanceCandidate",
    "LocalAnnotationHelper",
    "PrecedingChunkContext",
    "QWEN_PIPELINE_VERSION",
    "QWEN_PROMPT_VERSION",
    "QwenAnnotationAdapter",
    "build_qwen_prompt",
]
