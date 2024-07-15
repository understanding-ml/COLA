import numpy as np

from explainers import pshap
from xai_cola.counterfactual_explainer import CounterFactualExplainer
from xai_cola.ml_model import Model
from .base_attributor import Attributor
from xai_cola.cola.matching import CounterfactualOptimalTransportPolicy

EPSILON = 1e-20
SHAP_SAMPLE_SIZE = 10000
# SHAP_SAMPLE_SIZE = "auto"

class PSHAP(Attributor):
    def __init__(self, ml_model: Model, explainer: CounterFactualExplainer):
        super().__init__(ml_model, explainer)
        self.joint_probs = 0
        self.shape_sample_size = SHAP_SAMPLE_SIZE


    def calculate_varphi(self):
        cal_joint_probs = CounterfactualOptimalTransportPolicy(self.explainer)
        self.joint_probs = cal_joint_probs.compute_prob_matrix_of_factual_and_counterfactual()
        shap_values = pshap.JointProbabilityExplainer(self.ml_model).shap_values(
            self.x_factual ,
            self.x_counterfactual ,
            self.joint_probs,
            self.shape_sample_size,
        )

        varphi = convert_matrix_to_policy(shap_values)
        return varphi
    
def convert_matrix_to_policy(matrix):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P
