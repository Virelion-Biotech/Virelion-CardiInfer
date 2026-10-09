import math
import numpy as np
from cardiinfer.models import LikelihoodTerm, ArtifactRef
from cardiinfer.discrepancy import score_likelihood_term


def test_independent_error_budget_is_in_gaussian_likelihood_and_report():
    term = LikelihoodTerm(
        term_id="lat",
        model_output="lat",
        discrepancy="gaussian",
        observation_ref=ArtifactRef(artifact_id="o", kind="lat", uri="memory://obs"),
        noise_parameters={"sigma": 3, "model_discrepancy_sd": 4, "numerical_error_sd": 12},
    )
    score, detail = score_likelihood_term(term, np.array([13.0]), np.array([0.0]))
    assert detail["uncertainty_budget"]["effective_sd"] == 13
    assert abs(score - (0.5 * math.log(2 * math.pi) + math.log(13) + 0.5)) < 1e-12
