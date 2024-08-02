import numpy as np
from abc import ABC, abstractmethod

from xai_cola.data import BaseData
from xai_cola.ml_model import BaseModel
from xai_cola.counterfactual_explainer import CounterFactualExplainer

from xai_cola.cola import DataComposer # q
from xai_cola.cola import Attributor # varphi
from xai_cola.cola import BaseMatcher # joint_p

class Policy(ABC):
    def __init__(
            self, data:BaseData, ml_model:BaseModel,
            x_factual:np, x_counterfactual:np,
            p:BaseMatcher, varphi:Attributor, q:DataComposer
            #  explainer:CounterFactualExplainer, 
            ):
        
        self.data = data
        self.ml_model = ml_model
        # self.explainer = explainer
        self.x_factual = x_factual
        self.x_counterfactual = x_counterfactual

        self.p = p
        self.varphi = varphi
        self.q = q


        
    @abstractmethod
    def counterfactual_with_limited_actions(self):
        pass