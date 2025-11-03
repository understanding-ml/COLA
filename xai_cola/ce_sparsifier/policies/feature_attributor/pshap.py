import numpy as np
import shap
from xai_cola.ce_sparsifier.models import Model

from .base_attributor import Attributor

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"


class PSHAP(Attributor):
    def __init__(self, ml_model: Model, x_factual: np, x_counterfactual: np, joint_prob: np, random_state=42):
        super().__init__(ml_model, x_factual, x_counterfactual, joint_prob)
        self.shape_sample_size = SHAP_SAMPLE_SIZE
        self.random_state = random_state

    def calculate_varphi(self):
        shap_values = JointProbabilityExplainer(self.ml_model, random_state=self.random_state).shap_values(
            self.x_factual,
            self.x_counterfactual,
            self.joint_prob,
            shap_sample_size=self.shape_sample_size,
        )
        varphi = convert_matrix_to_policy(shap_values)
        return varphi


def convert_matrix_to_policy(matrix):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P



class WeightedExplainer:
    """
    This class provides explanations for model predictions using SHAP values,
    weighted according to a given probability distribution.
    """

    def __init__(self, model, random_state=42):
        """
        Initializes the WeightedExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to predict probabilities which are necessary
                      for SHAP value computation.
        :param random_state: Random seed for reproducible sampling. Default is 42.
        """
        self.model = model
        # Create a local random number generator for reproducibility
        self.rng = np.random.RandomState(random_state)

    def explain_instance(
        self, x, X_baseline, weights, sample_size=1000, shap_sample_size="auto"
    ):
        """
        Generates SHAP values for a single instance using a weighted sample of baseline data.

        :param x: The instance to explain. This should be a single data point.
        :param X_baseline: A dataset used as a reference or background distribution.
        :param weights: A numpy array of weights corresponding to the probabilities
                        of choosing each instance in X_baseline.
        :param num_samples: The number of samples to draw from X_baseline to create
                            the background dataset for the SHAP explainer.
        :return: An array of SHAP values for the instance.
        """
        # Normalize weights to ensure they sum to 1
        weights = weights + EPSILON
        weights = weights / (weights.sum())

        # Generate samples weighted by joint probabilities
        # Use local RNG for reproducibility
        indice = self.rng.choice(
            X_baseline.shape[0], p=weights, replace=True, size=sample_size
        )
        indice = np.unique(indice)
        sampled_X_baseline = X_baseline[indice]

        # Use the sampled_X_baseline as the background data for this specific explanation
        explainer_temp = shap.KernelExplainer(
            self.model.predict_proba, sampled_X_baseline
        )
        shap_values = explainer_temp.shap_values(x, nsamples=shap_sample_size)

        return shap_values


class JointProbabilityExplainer:
    """
    This class provides SHAP explanations for model predictions across multiple instances,
    using joint probability distributions to weight the baseline data for each instance.
    """

    def __init__(self, model, random_state=42):
        """
        Initializes the JointProbabilityExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to compute SHAP values using weighted baseline data.
        :param random_state: Random seed for reproducible sampling. Default is 42.
        """
        self.model = model
        self.weighted_explainer = WeightedExplainer(model, random_state=random_state)

    def shap_values(
        self, X, X_baseline, joint_probs, sample_size=1000, shap_sample_size="auto"
    ):
        """
        Computes SHAP values for multiple instances using a set of joint probability weights.

        :param X: An array of instances to explain. Each instance is a separate data point.
        :param X_baseline: A dataset used as a reference or background distribution.
        :param joint_probs: A matrix of joint probabilities, where each row corresponds to the
                            probabilities for an instance in X, used to weight X_baseline.
        :param num_samples: The number of samples to draw from X_baseline for each instance in X.
        :return: A numpy array of SHAP values for each instance in X.
        """
        return np.array(
            [
                self.weighted_explainer.explain_instance(
                    x,
                    X_baseline,
                    weights,
                    sample_size=sample_size,
                    shap_sample_size=shap_sample_size,
                )
                for x, weights in zip(X, joint_probs)
            ]
        )
