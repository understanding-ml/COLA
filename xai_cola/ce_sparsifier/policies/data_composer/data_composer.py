import numpy as np
import string

class DataComposer:
    """
    Data Composer class.
    
    Used to compute composed data based on joint probabilities and counterfactual data
    through different combination methods. Primarily used for data processing and 
    aggregation in counterfactual explanations.
    """
    def __init__(self, x_counterfactual:np, joint_prob:np, method:string):
        """
        Initialize the Data Composer.
        
        Args:
            x_counterfactual (np.ndarray): Counterfactual data matrix with shape (M, P),
                                          where M is the number of counterfactual samples
                                          and P is the feature dimension.
            joint_prob (np.ndarray): Joint probability matrix with shape (N, M),
                                    where N is the number of original samples
                                    and M is the number of counterfactual samples.
            method (str): Combination method, either "avg" (weighted average) or "max" (maximum weight).
        """
        self.x_counterfactual = x_counterfactual
        self.joint_probs = joint_prob
        self.method = method

    def calculate_q(self):
        """
        Calculate the composed data Q.
        
        Computes the combination result using the specified method based on 
        joint probabilities and counterfactual data.
        
        Returns:
            np.ndarray: Composed data matrix with shape (N, P),
                       where N is the number of original samples
                       and P is the feature dimension.
        """
        q = A_values(W=self.joint_probs, R=self.x_counterfactual, method=self.method)
        return q


def A_values(W, R, method):
    """
    Compute the composed result Q based on weight matrix W and data matrix R using the specified method.
    
    Args:
        W (np.ndarray): Weight matrix (joint probabilities) with shape (N, M),
                       where N is the number of original samples
                       and M is the number of counterfactual samples.
        R (np.ndarray): Data matrix (counterfactual data) with shape (M, P),
                       where M is the number of counterfactual samples
                       and P is the feature dimension.
        method (str): Combination method:
                     - "avg": Weighted average method. For each original sample,
                             uses normalized weights to compute a weighted average of counterfactual data.
                     - "max": Maximum weight method. For each original sample,
                             selects the counterfactual sample with the maximum weight.
    
    Returns:
        np.ndarray: Composed data matrix Q with shape (N, P),
                   where Q[i, :] represents the composed result for the i-th original sample.
    """
    N, M = W.shape  # N: number of original samples, M: number of counterfactual samples
    _, P = R.shape  # P: feature dimension
    Q = np.zeros((N, P))  # Initialize result matrix

    if method == "avg":
        # Weighted average method: for each original sample, use normalized weights
        # to compute weighted average of counterfactual data
        for i in range(N):
            weights = W[i, :]  # Get the weight vector for the i-th original sample
            # Normalize weights to ensure they sum to 1
            normalized_weights = weights / np.sum(weights)
            # Reshape for broadcasting with R: (M,) -> (M, 1)
            normalized_weights = normalized_weights.reshape(-1, 1)
            # Compute weighted sum: normalized_weights * R gives weighted data, then sum along axis=0
            Q[i, :] = np.sum(normalized_weights * R, axis=0)
    elif method == "max":
        # Maximum weight method: for each original sample, select the counterfactual sample with maximum weight
        for i in range(N):
            # Find the index of the counterfactual sample with maximum weight
            max_weight_index = np.argmax(W[i, :])
            # Use the counterfactual sample with maximum weight directly as the result
            Q[i, :] = R[max_weight_index, :]
    else:
        raise NotImplementedError(f"Unsupported method: {method}. Supported methods: 'avg', 'max'")
    return Q