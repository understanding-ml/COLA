import numpy as np
import shap
from copy import deepcopy

EPSILON = 1e-20


class WeightedExplainer:
    """
    This class provides explanations for model predictions using SHAP values,
    weighted according to a given probability distribution.
    """

    def __init__(self, model):
        """
        Initializes the WeightedExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to predict probabilities which are necessary
                      for SHAP value computation.
        """
        self.model = model

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
        indice = np.random.choice(
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

    def __init__(self, model):
        """
        Initializes the JointProbabilityExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to compute SHAP values using weighted baseline data.
        """
        self.model = model
        self.weighted_explainer = WeightedExplainer(model)

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
