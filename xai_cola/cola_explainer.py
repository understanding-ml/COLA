import string
import numpy as np

from xai_cola.data import BaseData, PandasData
from xai_cola.ml_model import BaseModel
from xai_cola.counterfactual_explainer import CounterFactualExplainer, DiCE

from xai_cola.cola.matching import BaseMatcher, CounterfactualOptimalTransportPolicy
from xai_cola.cola.data_composer import DataComposer
from xai_cola.cola.feature_attributor import Attributor, PSHAP
from xai_cola.counterfactual_refiner import Policy, PshapWithOTmatcher

class COunterfactualwithLimitedActions:
    def __init__(
            self, data: BaseData, ml_model: BaseModel,
            explainer: CounterFactualExplainer = None, x_factual: np = None, x_counterfactual: np = None, 
            matcher: string = None, attributor: string = None, Avalues_method: string = None, 
            limited_actions=None,
            ):
        
        self.data = data
        self.ml_model = ml_model
        self.explainer = explainer
        self.limited_actions = limited_actions
        self.matcher = matcher
        self.attributor = attributor
        self.Avalues_method = Avalues_method

        if explainer is not None:
            self.x_factual = explainer.factual
            self.x_counterfactual = explainer.counterfactual
        elif x_factual is not None and x_counterfactual is not None:
            self.x_factual = x_factual
            self.x_counterfactual = x_counterfactual
        else:
            raise ValueError("Either explainer or both x_factual and x_counterfactual must be provided")

        self.policy = None

    def generate_results(self):
        a = self.choose_policy()
        factual, ce, ace = a.counterfactual_with_limited_actions(self.limited_actions)
        return factual, ce, ace

    def get_matcher(self):
        if self.matcher == "ot":
            return CounterfactualOptimalTransportPolicy(self.x_factual, self.x_counterfactual).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "uniform":
            pass

    def get_attributor(self):
        if self.attributor == "pshap":
            varphi = PSHAP(ml_model=self.ml_model, x_factual=self.x_factual, x_counterfactual=self.x_counterfactual, joint_prob=self.get_matcher()).calculate_varphi()
            return varphi
        elif self.attributor == "randomshap":
            pass

    def get_data_composer(self):
        return DataComposer(x_factual=self.x_factual, x_counterfactual=self.x_counterfactual, joint_prob=self.get_matcher(), method=self.Avalues_method).calculate_q()

    def choose_policy(self):
        if self.matcher == "ot":
            p = self.get_matcher()
            if self.attributor == "pshap":
                varphi = self.get_attributor()
                q = self.get_data_composer()
                policy = PshapWithOTmatcher(
                    data=self.data, 
                    ml_model=self.ml_model,
                    x_factual=self.x_factual,
                    x_counterfactual=self.x_counterfactual,
                    p=p, 
                    varphi=varphi, 
                    q=q
                    )
                return policy
