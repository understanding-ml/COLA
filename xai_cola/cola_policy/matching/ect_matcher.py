import numpy as np
import pandas as pd

from .base_matcher import BaseMatcher

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"


class CounterfactualExactMatchingPolicy(BaseMatcher):
    def __init__(self, x_factual, x_counterfactual, prob_matrix=None):
        super().__init__(x_factual, x_counterfactual)
        self.prob_matrix = prob_matrix

    """Compute the prob_matrix of factual and counterfactual"""
    def compute_prob_matrix_of_factual_and_counterfactual(self):

        num_instances_factual = self.x_factual.shape[0]
        num_instances_counterfactual = self.x_counterfactual.shape[0]
        
        if num_instances_factual == num_instances_counterfactual:
            prob_matrix = np.eye(num_instances_factual)      
            prob_matrix = convert_matrix_to_policy(prob_matrix)  # Normalize to make the sum of all elements 1
        else:   # ask user to provide the matrix of their own probabilities
            raise ValueError("The number of instances in factual and counterfactual data must be the same.")
            pass
        return prob_matrix
    
def convert_matrix_to_policy(prob_matrix):
    P = np.abs(prob_matrix) / np.abs(prob_matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P