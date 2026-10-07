"""Run outside the checkout against the installed wheel, using a real command."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cardiinfer
from cardiinfer import (
    ArtifactRef,
    InferenceRequest,
    LikelihoodTerm,
    NativeMetropolisBackend,
    ParameterPrior,
    UncertaintyPropagationRequest,
)
from cardiinfer.provenance import local_file_path, strict_loads


def main() -> None:
    assert cardiinfer.__version__ == "0.5.0"
    assert "site-packages" in str(Path(cardiinfer.__file__).resolve())
    with TemporaryDirectory() as directory:
        root = Path(directory)
        forward = root / "forward.py"
        forward.write_text(
            "import json, os\n"
            "x = json.loads(os.environ['HEARTTWIN_PAYLOAD'])['parameters']['x']\n"
            "print(json.dumps({'outputs': {'y': [2*x+1]}}))\n",
            encoding="utf-8",
        )
        context = {"forward_model": {"mode": "command", "command": [sys.executable, str(forward)]}}
        backend = NativeMetropolisBackend()
        result = backend.infer(
            InferenceRequest(
                subject_id="wheel-smoke",
                backend=backend.name,
                model_service="Toy",
                model_capability="toy.simulate",
                priors=[ParameterPrior(name="x", distribution="uniform", bounds=(0, 1))],
                likelihood=[
                    LikelihoodTerm(
                        term_id="y",
                        observation_ref=ArtifactRef(
                            artifact_id="inline", kind="synthetic", uri="file:///unused"
                        ),
                        model_output="outputs.y",
                        discrepancy="gaussian",
                        noise_parameters={"sigma": 0.05},
                        metadata={"observed": [1.6]},
                    )
                ],
                model_context=context,
                sampler_settings={
                    "n_chains": 2,
                    "warmup": 20,
                    "draws": 40,
                    "proposal_scale": 0.1,
                    "output_dir": str(root),
                },
                seed=12,
            )
        )
        assert result.posterior_samples is not None
        strict_loads(local_file_path(result.posterior_samples.uri).read_text(encoding="utf-8"))
        assert abs(result.posterior[0].mean - 0.3) < 0.08
        propagated = backend.propagate(
            UncertaintyPropagationRequest(
                subject_id=result.subject_id,
                backend=backend.name,
                model_service="Toy",
                model_capability="toy.simulate",
                posterior_samples=result.posterior_samples,
                outputs=["outputs.y"],
                model_context=context,
                settings={"max_samples": 5, "output_dir": str(root)},
            )
        )
        assert "outputs.y" in propagated.output_summaries
    print("Installed-wheel command inference, posterior artifact and propagation passed")


if __name__ == "__main__":
    main()
