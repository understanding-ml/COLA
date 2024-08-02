import numpy
from abc import ABC, abstractmethod

from xai_cola.counterfactual_explainer import CounterFactualExplainer

class BaseMatcher(ABC):
    def __init__(self, x_factual:numpy, x_counterfactual:numpy):
        super().__init__()

        self.x_factual = x_factual
        self.x_counterfactual = x_counterfactual
        self.N = x_factual.shape[0]
        self.M = x_counterfactual.shape[0]


    @abstractmethod
    def compute_prob_matrix_of_factual_and_counterfactual(self):
        pass
