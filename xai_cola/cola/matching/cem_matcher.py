import ot
import numpy as np
import pandas as pd
from cem import CEM
from .base_matcher import BaseMatcher
from xai_cola.counterfactual_explainer import CounterFactualExplainer

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"

class CounterfactualCoarsenedExactMatchingOTPolicy(BaseMatcher):
    def __init__(self,x_factual, x_counterfactual, n_bins=5):
        super().__init__(x_factual, x_counterfactual)
        self.n_bins = n_bins

    def _compute_cem_prob_matrix(self, x, r):
        # Combine the factual and counterfactual data
        combined_data = np.vstack((x, r))
        treatment_indicator = np.hstack((np.zeros(x.shape[0]), np.ones(r.shape[0])))

        # Perform Coarsened Exact Matching
        df = pd.DataFrame(combined_data)
        df["treatment"] = treatment_indicator
        cem_result = CEM(df, "treatment", drop="drop", cut=self.n_bins)
        matched_groups = cem_result["matched"]

        # Initialize the probability matrix
        prob_matrix = np.zeros((x.shape[0], r.shape[0]))

        # Fill the probability matrix based on matching results
        for group in matched_groups:
            x_indices = [idx for idx in group if idx < x.shape[0]]
            r_indices = [idx - x.shape[0] for idx in group if idx >= x.shape[0]]
            if x_indices and r_indices:
                prob = 1.0 / len(x_indices)
                for x_idx in x_indices:
                    for r_idx in r_indices:
                        prob_matrix[x_idx, r_idx] = prob
        return prob_matrix

    def _compute_ot_for_unmatched(self, x, r, prob_matrix):
        # Identify unmatched rows in x
        row_sums = prob_matrix.sum(axis=1)
        unmatched_x_indices = np.where(row_sums == 0)[0]

        if len(unmatched_x_indices) == 0:
            return prob_matrix

        # Compute the cost matrix for unmatched rows
        cost_matrix = ot.dist(x[unmatched_x_indices], r, metric="euclidean")

        # Uniform distribution over the unmatched rows of x and all rows of r
        a = np.ones(len(unmatched_x_indices)) / len(unmatched_x_indices)
        b = np.ones(r.shape[0]) / r.shape[0]

        # Compute the optimal transport plan
        transport_plan = ot.emd(a, b, cost_matrix)

        # Update the probability matrix with the OT results for unmatched rows
        for i, x_idx in enumerate(unmatched_x_indices):
            for j, r_idx in enumerate(range(r.shape[0])):
                prob_matrix[x_idx, r_idx] = transport_plan[i, j]

        return prob_matrix

    """Compute the prob_matrix of factual and counterfactual"""
    def compute_prob_matrix_of_factual_and_counterfactual(self):
        # Compute initial probability matrix using CEM
        p = self._compute_cem_prob_matrix(self.x_factual, self.x_counterfactual)

        # Ensure all rows in x are matched using OT for unmatched rows
        p = self._compute_ot_for_unmatched(self.x_factual, self.x_counterfactual, p)
        return p
