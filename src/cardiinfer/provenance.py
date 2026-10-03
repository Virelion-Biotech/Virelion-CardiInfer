from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    actual = file_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Artifact SHA-256 mismatch for {Path(path)}: "
            f"expected {expected_sha256.lower()}, got {actual.lower()}"
        )


def write_json_artifact(
    output_dir: str | Path,
    *,
    artifact_id: str,
    kind: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
):
    from .models import ArtifactRef

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{artifact_id}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    digest = file_sha256(path)
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.as_uri(),
        sha256=digest,
        metadata=dict(metadata or {}),
    )
