from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.as_uri(),
        sha256=digest,
        metadata=dict(metadata or {}),
    )
