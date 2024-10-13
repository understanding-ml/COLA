import numpy as np
import string

class DataComposer:
    def __init__(self, x_counterfactual:np, joint_prob:np, method:string):
        self.x_counterfactual = x_counterfactual
        self.joint_probs = joint_prob
        self.method = method

    def calculate_q(self):
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