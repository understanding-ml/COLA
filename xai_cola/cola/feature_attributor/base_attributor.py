from abc import ABC, abstractmethod
from xai_cola.ml_model import Model
from xai_cola.counterfactual_explainer import CounterFactualExplainer

class Attributor(ABC):
    def __init__(self, ml_model: Model, explainer: CounterFactualExplainer):
        self.ml_model = ml_model.load_model()
        self.x_factual = explainer.factual
        self.x_counterfactual = explainer.counterfactual
        self.explainer = explainer
    
    @abstractmethod
    def calculate_varphi(self):
        pass