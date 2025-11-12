import ot
import numpy as np

from .base_matcher import BaseMatcher

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"

class CounterfactualOptimalTransportPolicy(BaseMatcher):
    def __init__(self, x_factual: np, x_counterfactual: np):
        super().__init__(x_factual, x_counterfactual)

        self.reg = 0   ## should input the reg 
        # self.method = method
        self.ot_cost = ot.dist(self.x_factual, self.x_counterfactual, p=2)
        self.probs_matrix = 0

    
    """Compute the prob_matrix of factual and counterfactual"""
    def compute_prob_matrix_of_factual_and_counterfactual(self):
        if self.reg <= 0:
            self.probs_matrix = ot.emd(
                np.ones(self.N) / self.N, np.ones(self.M) / self.M, self.ot_cost
            )
        else:
            self.probs_matrix = ot.bregman.sinkhorn(
                np.ones(self.N) / self.N,
                np.ones(self.M) / self.M,
                self.ot_cost,
                reg=self.reg,
                numItermax=5000,
            )
        ot_matcher = self.convert_matrix_to_policy()
        return ot_matcher   

    def convert_matrix_to_policy(self):
        P = np.abs(self.probs_matrix) / np.abs(self.probs_matrix).sum()
        P += EPSILON
        P /= P.sum()
        return P