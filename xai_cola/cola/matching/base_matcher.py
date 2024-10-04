import numpy as np
from abc import ABC, abstractmethod

from xai_cola.counterfactual_explainer import CounterFactualExplainer

EPSILON = 1e-6

class BaseMatcher(ABC):
    def __init__(self, x_factual:np, x_counterfactual:np):
        super().__init__()

        self.x_factual = x_factual
        self.x_counterfactual = x_counterfactual
        self.N = x_factual.shape[0]
        self.M = x_counterfactual.shape[0]

    """Compute the prob_matrix of factual and counterfactual"""
    @abstractmethod
    def compute_prob_matrix_of_factual_and_counterfactual(self):
        pass

    def convert_matrix_to_policy(self):
        P = np.abs(self.probs_matrix) / np.abs(self.probs_matrix).sum()
        P += EPSILON
        P /= P.sum()
        return P