from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .models import ArtifactRef, LikelihoodTerm
from .provenance import local_file_path, strict_loads


def extract_path(payload: Any, path: str) -> Any:
    if isinstance(payload, dict) and path in payload:
        return payload[path]
    current = payload
    for token in path.split("."):
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(path)
            current = current[token]
        elif isinstance(current, (list, tuple)) and token.isdigit():
            current = current[int(token)]
        else:
            raise KeyError(path)
    return current


def _artifact_path(ref: ArtifactRef) -> Path:
    return local_file_path(ref.uri)


def load_artifact(ref: ArtifactRef) -> Any:
    path = _artifact_path(ref)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if ref.sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != ref.sha256.lower():
            raise ValueError(
                f"Artifact digest mismatch for {ref.artifact_id}: expected {ref.sha256}, got {digest}"
            )
    return strict_loads(data.decode("utf-8"))


def _numeric(value: Any, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} is not numeric") from exc
    if array.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def resolve_observed(term: LikelihoodTerm) -> np.ndarray:
    if "observed" in term.metadata:
        return _numeric(term.metadata["observed"], label=f"{term.term_id} observed data")

    payload = load_artifact(term.observation_ref)
    if "observation_path" in term.metadata:
        return _numeric(
            extract_path(payload, str(term.metadata["observation_path"])),
            label=f"{term.term_id} observed data",
        )
    if isinstance(payload, list):
        return _numeric(payload, label=f"{term.term_id} observed data")
    if isinstance(payload, dict):
        candidates = [
            term.model_output,
            "values",
            "data",
            "signal",
        ]
        for key in candidates:
            try:
                value = extract_path(payload, key)
            except KeyError:
                continue
            return _numeric(value, label=f"{term.term_id} observed data")
        outputs = payload.get("outputs")
        if isinstance(outputs, dict) and term.model_output in outputs:
            return _numeric(outputs[term.model_output], label=f"{term.term_id} observed data")
    raise KeyError(
        f"Could not resolve observed data for likelihood term {term.term_id!r}; "
        "set metadata.observation_path or metadata.observed"
    )


def resolve_predicted(output: Any, term: LikelihoodTerm) -> np.ndarray:
    path = str(term.metadata.get("model_output_path") or term.model_output)
    try:
        value = extract_path(output, path)
    except KeyError:
        if isinstance(output, dict) and isinstance(output.get("outputs"), dict):
            value = extract_path(output["outputs"], path)
        else:
            raise KeyError(
                f"Forward output does not contain model output {path!r} for term {term.term_id!r}"
            ) from None
    return _numeric(value, label=f"{term.term_id} predicted data")


def _aligned(
    predicted: np.ndarray, observed: np.ndarray, term: LikelihoodTerm
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    obs = np.asarray(observed, dtype=float).reshape(-1)
    if pred.size != obs.size:
        if str(term.metadata.get("alignment", "strict")) != "truncate":
            raise ValueError(
                f"Likelihood term {term.term_id!r} has {pred.size} predicted values "
                f"but {obs.size} observed values"
            )
        size = min(pred.size, obs.size)
        pred = pred[:size]
        obs = obs[:size]
    if pred.size == 0:
        raise ValueError(f"Likelihood term {term.term_id!r} has no aligned samples")
    return pred, obs


def score_likelihood_term(
    term: LikelihoodTerm,
    predicted: np.ndarray,
    observed: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    pred, obs = _aligned(
        _numeric(predicted, label="predicted data"), _numeric(observed, label="observed data"), term
    )
    residual = pred - obs
    method = term.discrepancy

    if method == "rmse":
        score = float(np.sqrt(np.mean(residual**2)))
    elif method == "mae":
        score = float(np.mean(np.abs(residual)))
    elif method == "normalized_rmse":
        scale = float(np.std(obs))
        if scale <= 1e-12:
            scale = max(float(np.ptp(obs)), float(np.mean(np.abs(obs))), 1.0)
        score = float(np.sqrt(np.mean(residual**2)) / scale)
    elif method == "correlation":
        if np.allclose(pred, pred[0]) or np.allclose(obs, obs[0]):
            score = 0.0 if np.allclose(pred, obs) else 1.0
        else:
            corr = float(np.corrcoef(pred, obs)[0, 1])
            score = float(1.0 - np.clip(corr, -1.0, 1.0))
    elif method == "cosine":
        denom = float(np.linalg.norm(pred) * np.linalg.norm(obs))
        if denom <= 1e-15:
            score = 0.0 if np.allclose(pred, obs) else 1.0
        else:
            score = float(1.0 - np.clip(np.dot(pred, obs) / denom, -1.0, 1.0))
    elif method == "huber":
        delta = float(term.noise_parameters.get("delta", 1.0))
        if delta <= 0:
            raise ValueError("Huber delta must be > 0")
        absolute = np.abs(residual)
        loss = np.where(absolute <= delta, 0.5 * residual**2, delta * (absolute - 0.5 * delta))
        score = float(np.mean(loss))
    elif method == "gaussian":
        sigma = float(term.noise_parameters.get("sigma", term.noise_parameters.get("sd", 1.0)))
        if sigma <= 0:
            raise ValueError("Gaussian likelihood sigma must be > 0")
        score = float(
            np.sum(0.5 * math.log(2.0 * math.pi) + math.log(sigma) + 0.5 * (residual / sigma) ** 2)
        )
    elif method == "student_t":
        df = float(term.noise_parameters.get("df", 4.0))
        scale = float(
            term.noise_parameters.get(
                "scale",
                term.noise_parameters.get("sigma", term.noise_parameters.get("sd", 1.0)),
            )
        )
        if df <= 0 or scale <= 0:
            raise ValueError("Student-t likelihood requires df > 0 and scale > 0")
        constant = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi)
            - math.log(scale)
        )
        logp = constant - 0.5 * (df + 1.0) * np.log1p((residual / scale) ** 2 / df)
        score = float(-np.sum(logp))
    else:
        raise ValueError(
            f"Native CardiInfer does not implement discrepancy {term.discrepancy!r}; "
            "provide a plugin backend for custom discrepancies"
        )

    weighted = float(term.weight * score)
    if not math.isfinite(weighted):
        raise ValueError("Likelihood/discrepancy exceeded finite numerical range")
    detail = {
        "term_id": term.term_id,
        "model_output": term.model_output,
        "discrepancy": term.discrepancy,
        "weight": float(term.weight),
        "score": score,
        "weighted_score": weighted,
        "n": int(pred.size),
        "residual_mean": float(np.mean(residual)),
        "residual_sd": float(np.std(residual)),
    }
    return weighted, detail
