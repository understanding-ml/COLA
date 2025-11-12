import numpy as np

class DataComposer:
    """
    Data Composer class.
    
    Used to compute composed data based on joint probabilities and counterfactual data
    through different combination methods. Primarily used for data processing and 
    aggregation in counterfactual explanations.
    """
    def __init__(self, x_counterfactual:np, joint_prob:np):
        """
        Initialize the Data Composer.

        Args:
            x_counterfactual (np.ndarray): Counterfactual data matrix with shape (M, P),
                                          where M is the number of counterfactual samples
                                          and P is the feature dimension.
            joint_prob (np.ndarray): Joint probability matrix with shape (N, M),
                                    where N is the number of original samples
                                    and M is the number of counterfactual samples.
        """
        self.x_counterfactual = x_counterfactual
        self.joint_probs = joint_prob

    def calculate_q(self):
        """
        Calculate the composed data q.

        Computes the combination result using the max method based on
        joint probabilities and counterfactual data.

        Returns:
            np.ndarray: Composed data matrix with shape (N, P),
                       where N is the number of original samples
                       and P is the feature dimension.
                       Each row of Q corresponds exactly to a row in x_counterfactual.
        """
        q = A_values(W=self.joint_probs, R=self.x_counterfactual)
        return q


def A_values(W, R):
    """
    Compute the composed result Q based on weight matrix W and data matrix R using the max method.

    For each original sample, selects the counterfactual sample with the maximum weight.
    This ensures that each row in Q is an exact copy of a corresponding row from R (x_counterfactual).

    Args:
        W (np.ndarray): Weight matrix (joint probabilities) with shape (N, M),
                       where N is the number of original samples
                       and M is the number of counterfactual samples.
        R (np.ndarray): Data matrix (counterfactual data) with shape (M, P),
                       where M is the number of counterfactual samples
                       and P is the feature dimension.

    Returns:
        np.ndarray: Composed data matrix Q with shape (N, P),
                   where Q[i, :] is an exact copy of R[argmax(W[i, :]), :].
                   Each row of Q corresponds exactly to a row in R with identical values.
    """
    N, M = W.shape  # N: number of original samples, M: number of counterfactual samples
    _, P = R.shape  # P: feature dimension
    # Initialize result matrix with the same dtype as R to preserve data types (e.g., strings, integers)
    Q = np.empty((N, P), dtype=R.dtype)

    # Maximum weight method: for each original sample, select the counterfactual sample with maximum weight
    for i in range(N):
        # Find the index of the counterfactual sample with maximum weight
        max_weight_index = np.argmax(W[i, :])
        # Use the counterfactual sample with maximum weight directly as the result
        # This ensures Q[i, :] is an exact copy of R[max_weight_index, :]
        Q[i, :] = R[max_weight_index, :]

    return Q