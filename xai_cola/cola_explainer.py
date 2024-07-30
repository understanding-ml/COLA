import string

from xai_cola.data import BaseData, PandasData
from xai_cola.ml_model import BaseModel
from xai_cola.counterfactual_explainer import CounterFactualExplainer, DiCE

from xai_cola.cola.matching import BaseMatcher, CounterfactualOptimalTransportPolicy
from xai_cola.cola.data_composer import DataComposer
from xai_cola.cola.feature_attributor import Attributor, PSHAP
from xai_cola.counterfactual_refiner import Policy, PshapWithOTmatcher

class COunterfactualwithLimitedActions():
    def __init__(self, data:BaseData, ml_model:BaseModel, explainer:CounterFactualExplainer, limited_actions, matcher:string, attributor:string, Avalues_method:string):
        self.data = data
        self.ml_model = ml_model
        self.explainer = explainer
        self.limited_actions = limited_actions
        self.matcher = matcher
        self.attributor = attributor
        self.Avalues_method = Avalues_method

        self.x_factual = explainer.factual
        self.x_counterfactual = explainer.counterfactual
        self.policy = None

    def generate_results(self):
        a = self.choose_policy()
        factual, ce, ace = a.counterfactual_with_limited_actions(self.limited_actions)
        return factual, ce, ace

    def get_matcher(self):
        if self.matcher == "ot":
            # print(f"self.explainer ={self.explainer.x_factual}")
            return CounterfactualOptimalTransportPolicy(self.explainer).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "uniform":
            pass


    # get the varphi
    def get_attributor(self):
        if self.attributor == "pshap":
            varphi = PSHAP(ml_model=self.ml_model, explainer=self.explainer).calculate_varphi()
            return varphi
        elif self.attributor == "randomshap":
            pass

    # get the q
    def get_data_composer(self):
        return DataComposer(explainer=self.explainer, method=self.Avalues_method).calculate_q()

    def choose_policy(self):  # Pshap（feature attributor） with optimal matcher (matching: joint probs)  with datacomposer(max/avg)
        if self.matcher == "ot":
            p = self.get_matcher()
            if self.attributor == "pshap":
                varphi = self.get_attributor()
                q = self.get_data_composer()
                policy = PshapWithOTmatcher(data=self.data, ml_model=self.ml_model, explainer= self.explainer, p=p, varphi=varphi, q=q)
                return policy