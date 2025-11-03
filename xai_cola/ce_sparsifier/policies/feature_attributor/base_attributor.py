import numpy as np
from abc import ABC, abstractmethod

from xai_cola.ce_sparsifier.models import Model

class Attributor(ABC):
    def __init__(self, ml_model: Model, x_factual:np, x_counterfactual:np, joint_prob:np):
        self.ml_model = ml_model
        self.x_factual = x_factual
        self.x_counterfactual = x_counterfactual
        self.joint_prob = joint_prob
    
    @abstractmethod
    def calculate_varphi(self):
        pass