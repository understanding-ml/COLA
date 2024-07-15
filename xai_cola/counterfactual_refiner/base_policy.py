from abc import ABC, abstractmethod

from xai_cola.data import BaseData
from xai_cola.ml_model import BaseModel
from xai_cola.counterfactual_explainer import CounterFactualExplainer

from xai_cola.cola import DataComposer # q
from xai_cola.cola import Attributor # varphi
from xai_cola.cola import BaseMatcher # joint_p

class Policy(ABC):
    def __init__(self, data:BaseData, ml_model:BaseModel, explainer:CounterFactualExplainer, p:BaseMatcher, varphi:Attributor, q:DataComposer):
        self.p = p
        self.varphi = varphi
        self.q = q
        self.explainer = explainer
        self.x_factual = explainer.factual
        self.x_counterfactual = explainer.counterfactual
        self.data = data
        self.ml_model = ml_model
        
    @abstractmethod
    def counterfactual_with_limited_actions(self):
        pass