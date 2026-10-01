"""EP-specific inference builders for ElectroTrace/CardiEP integration."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import ArtifactRef, InferenceRequest, LikelihoodTerm, ParameterPrior


def _observation_artifacts(
    handoff: Mapping[str, Any],
) -> tuple[dict[str, ArtifactRef], dict[str, str]]:
    raw = handoff.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError("EP measurement handoff requires a non-empty observations array")
    artifacts: dict[str, ArtifactRef] = {}
    kinds: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("Each EP observation must be an object")
        observation_id = str(item.get("observation_id") or "")
        if not observation_id:
            raise ValueError("Each EP observation requires observation_id")
        if observation_id in artifacts:
            raise ValueError(f"Duplicate EP observation_id: {observation_id}")
        artifacts[observation_id] = ArtifactRef.model_validate(item.get("artifact"))
        kinds[observation_id] = str(item.get("kind") or "other")
    return artifacts, kinds


def likelihood_from_electrotrace(
    handoff: Mapping[str, Any],
) -> list[LikelihoodTerm]:
    """Turn ElectroTrace likelihood hints into CardiInfer likelihood terms.

    Hints remain advisory until they are validated here.  The observation
    artifact itself is the authoritative measured-data reference.
    """
    artifacts, kinds = _observation_artifacts(handoff)
    hints = handoff.get("likelihood_hints")
    terms: list[LikelihoodTerm] = []

    if isinstance(hints, list) and hints:
        for index, hint in enumerate(hints):
            if not isinstance(hint, Mapping):
                raise TypeError("likelihood_hints must contain objects")
            observation_id = hint.get("observation_id")
            if observation_id is None:
                term_id = str(hint.get("term_id") or "")
                prefix = term_id.split(":", 1)[0]
                if prefix in artifacts:
                    observation_id = prefix
                elif len(artifacts) == 1:
                    observation_id = next(iter(artifacts))
            observation_id = str(observation_id or "")
            if observation_id not in artifacts:
                raise ValueError(
                    f"Likelihood hint {index} does not identify a known observation"
                )
            terms.append(
                LikelihoodTerm(
                    term_id=str(
                        hint.get("term_id")
                        or f"{observation_id}:{hint.get('model_output', 'output')}"
                    ),
                    observation_ref=artifacts[observation_id],
                    model_output=str(hint.get("model_output") or "ecg"),
                    discrepancy=str(hint.get("discrepancy") or "gaussian"),
                    weight=float(hint.get("weight", 1.0)),
                    noise_parameters={
                        str(k): float(v)
                        for k, v in dict(hint.get("noise_parameters") or {}).items()
                    },
                    metadata={
                        "observation_id": observation_id,
                        **dict(hint.get("metadata") or {}),
                    },
                )
            )
    else:
        for observation_id, artifact in artifacts.items():
            kind = kinds[observation_id]
            if kind == "ecg":
                model_output, discrepancy, weight = "ecg", "correlation", 1.0
            elif kind in {"eam_activation", "activation_map"}:
                model_output, discrepancy, weight = "activation_map", "student_t", 2.0
            elif kind == "repolarization_map":
                model_output, discrepancy, weight = "repolarization_map", "student_t", 2.0
            else:
                model_output, discrepancy, weight = "electrical_output", "gaussian", 1.0
            terms.append(
                LikelihoodTerm(
                    term_id=f"{observation_id}:{model_output}",
                    observation_ref=artifact,
                    model_output=model_output,
                    discrepancy=discrepancy,
                    weight=weight,
                    metadata={"observation_id": observation_id},
                )
            )

    ids = [term.term_id for term in terms]
    if len(ids) != len(set(ids)):
        raise ValueError("EP likelihood term IDs must be unique")
    return terms


def ep_inference_request_from_electrotrace(
    handoff: Mapping[str, Any],
    *,
    subject_id: str,
    inference_backend: str,
    ep_backend: str,
    anatomy_ref: Mapping[str, Any] | ArtifactRef,
    priors: Sequence[ParameterPrior | Mapping[str, Any]],
    ep_settings: Mapping[str, Any] | None = None,
    sampler_settings: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> InferenceRequest:
    """Build the CardiInfer problem that calibrates CardiEP to ElectroTrace data."""
    prior_models = [
        item if isinstance(item, ParameterPrior) else ParameterPrior.model_validate(item)
        for item in priors
    ]
    anatomy = (
        anatomy_ref
        if isinstance(anatomy_ref, ArtifactRef)
        else ArtifactRef.model_validate(anatomy_ref)
    )
    observations = list(handoff.get("observations") or [])
    return InferenceRequest(
        subject_id=subject_id,
        model_service="CardiEP",
        model_capability="ep.simulate",
        backend=inference_backend,
        priors=prior_models,
        likelihood=likelihood_from_electrotrace(handoff),
        model_context={
            "ep_backend": ep_backend,
            "anatomy_ref": anatomy.model_dump(mode="json"),
            "ep_observations": observations,
            "measurement_handoff_schema": handoff.get("schema_version"),
            "ep_settings": dict(ep_settings or {}),
        },
        sampler_settings=dict(sampler_settings or {}),
        seed=seed,
    )
