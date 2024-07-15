from abc import ABC, abstractmethod
from xai_cola.counterfactual_explainer import CounterFactualExplainer

class BaseMatcher(ABC):
    def __init__(self, explainer:CounterFactualExplainer):
        super().__init__()
        # self.model = model.load_model()

        self.x_factual = explainer.factual
        self.x_counterfactual = explainer.counterfactual
        self.N = explainer.factual.shape[0]
        # print(f"self.M = {explainer.x_counterfactual}")
        self.M = explainer.counterfactual.shape[0]

    @abstractmethod
    def compute_prob_matrix_of_factual_and_counterfactual(self):
        pass



        
#     class Policy:
#     def __init__(self, model, X_factual, X_counterfactual):
#         self.model = model
#         self.X_factual = X_factual
#         self.X_counterfactual = X_counterfactual
#         self.N, self.M = self.X_factual.shape[0], self.X_counterfactual.shape[0]


# class CounterfactualPolicy(Policy):
#     def __init__(self, model, X_factual, X_counterfactual):
#         super().__init__(model, X_factual, X_counterfactual)