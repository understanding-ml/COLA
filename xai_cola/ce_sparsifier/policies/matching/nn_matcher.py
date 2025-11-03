import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base_matcher import BaseMatcher

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"


class CounterfactualNearestNeighborMatchingPolicy(BaseMatcher):
    def __init__(self, x_factual, x_counterfactual):
        super().__init__(x_factual, x_counterfactual)

    """Compute the prob_matrix of factual and counterfactual"""
    def compute_prob_matrix_of_factual_and_counterfactual(self):
        # Fit nearest neighbors model on the counterfactual data
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.x_counterfactual)

        # Find the nearest neighbors in r for each row in x
        distances, indices = nn.kneighbors(self.x_factual)

        # Initialize the probability matrix
        prob_matrix = np.zeros((self.x_factual.shape[0], self.x_counterfactual.shape[0]))

        # Fill the probability matrix based on nearest neighbors
        for i, neighbor_index in enumerate(indices.flatten()):
            prob_matrix[i, neighbor_index] = 1.0

        return prob_matrix


