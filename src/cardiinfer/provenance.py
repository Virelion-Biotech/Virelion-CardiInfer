from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname


def canonical_json(value: Any) -> str:
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for nested in item.values():
                check(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                check(nested)

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def local_file_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError("Local file URIs cannot include remote hosts, query or fragment")
        return Path(url2pathname(parsed.path)).expanduser().resolve()
    if parsed.scheme and not (len(parsed.scheme) == 1 and uri[1:2] == ":"):
        raise ValueError("Artifact URI must be a local file URI or path")
    return Path(uri).expanduser().resolve()


def strict_loads(text: str | bytes) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"Duplicate JSON key: {key}")
            output[key] = value
        return output

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    canonical_json(value)  # Also rejects overflowed JSON numeric literals.
    return value


def write_json(path: str | Path, payload: Any) -> None:
    canonical_json(payload)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, newline=""
        ) as handle:
            name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


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

    if not artifact_id or artifact_id in {".", ".."} or any(ch in artifact_id for ch in "/\\"):
        raise ValueError("Artifact identifiers must be safe file names")
    root = Path(output_dir).expanduser().resolve()
    path = root / f"{artifact_id}.json"
    write_json(path, payload)
    digest = file_sha256(path)
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.as_uri(),
        sha256=digest,
        metadata=dict(metadata or {}),
    )
