import numpy as np
from xai_cola.counterfactual_explainer import CounterFactualExplainer
from xai_cola.cola.matching import CounterfactualOptimalTransportPolicy

class DataComposer:
    def __init__(self, explainer:CounterFactualExplainer, method):
        self.explainer = explainer
        self.method = method

        self.x_factual = explainer.factual
        self.x_counterfactual = explainer.counterfactual
        self.joint_probs = 0

    def calculate_q(self):
        counterfactual_explainer = CounterfactualOptimalTransportPolicy(self.explainer)
        self.joint_probs = counterfactual_explainer.compute_prob_matrix_of_factual_and_counterfactual()
        q = A_values(W=self.joint_probs, R=self.x_counterfactual, method=self.method)
        return q
 

def A_values(W, R, method):
    N, M = W.shape
    _, P = R.shape
    Q = np.zeros((N, P))

    if method == "avg":
        for i in range(N):
            weights = W[i, :]
            # Normalize weights to ensure they sum to 1
            normalized_weights = weights / np.sum(weights)
            # Reshape to match R's rows for broadcasting
            normalized_weights = normalized_weights.reshape(-1, 1)
            # Compute the weighted sum
            Q[i, :] = np.sum(normalized_weights * R, axis=0)
    elif method == "max":
        for i in range(N):
            max_weight_index = np.argmax(W[i, :])
            Q[i, :] = R[max_weight_index, :]
    else:
        raise NotImplementedError
    return Q