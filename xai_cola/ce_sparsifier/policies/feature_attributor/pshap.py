import numpy as np
import pandas as pd
import shap
from xai_cola.ce_sparsifier.models import Model

from .base_attributor import Attributor

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"


class PSHAP(Attributor):
    def __init__(self, ml_model: Model, x_factual: np, x_counterfactual: np, joint_prob: np, random_state=42, feature_names=None):
        super().__init__(ml_model, x_factual, x_counterfactual, joint_prob)
        self.shape_sample_size = SHAP_SAMPLE_SIZE
        self.random_state = random_state
        self.feature_names = feature_names

    def calculate_varphi(self):
        shap_values = JointProbabilityExplainer(
            self.ml_model,
            random_state=self.random_state,
            feature_names=self.feature_names
        ).shap_values(
            self.x_factual,
            self.x_counterfactual,
            self.joint_prob,
            shap_sample_size=self.shape_sample_size,
        )
        # print(f"[DEBUG PSHAP] shap_values shape after JointProbabilityExplainer: {shap_values.shape}")
        # print(f"[DEBUG PSHAP] shap_values type: {type(shap_values)}")
        varphi = convert_matrix_to_policy(shap_values)
        # print(f"[DEBUG PSHAP] varphi shape after convert_matrix_to_policy: {varphi.shape}")
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

    def __init__(self, model, random_state=42, feature_names=None):
        """
        Initializes the WeightedExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to predict probabilities which are necessary
                      for SHAP value computation.
        :param random_state: Random seed for reproducible sampling. Default is 42.
        :param feature_names: List of feature names to use when creating DataFrames.
        """
        self.model = model
        self.feature_names = feature_names
        # Create a local random number generator for reproducibility
        self.rng = np.random.RandomState(random_state)

    def _predict_proba_wrapper(self, X):
        """
        Wrapper for predict_proba that converts numpy arrays to DataFrames.
        This prevents sklearn warnings about missing feature names.
        """
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_proba(X)

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

        # Convert to DataFrame if feature names are available
        if self.feature_names is not None:
            sampled_X_baseline = pd.DataFrame(sampled_X_baseline, columns=self.feature_names)
            if x.ndim == 1:
                x = pd.DataFrame([x], columns=self.feature_names)
            else:
                x = pd.DataFrame(x, columns=self.feature_names)

        # Use the sampled_X_baseline as the background data for this specific explanation
        explainer_temp = shap.KernelExplainer(
            self._predict_proba_wrapper, sampled_X_baseline
        )
        shap_values = explainer_temp.shap_values(x, nsamples=shap_sample_size)

        # print(f"[DEBUG WeightedExplainer] Raw shap_values type: {type(shap_values)}")
        # if isinstance(shap_values, list):
        #     print(f"[DEBUG WeightedExplainer] shap_values is a list with {len(shap_values)} elements")
        #     for i, sv in enumerate(shap_values):
        #         print(f"[DEBUG WeightedExplainer]   Element {i} shape: {sv.shape}")
        # else:
        #     print(f"[DEBUG WeightedExplainer] shap_values shape: {shap_values.shape}")

        # Handle multi-class output: shap_values might be a list of arrays (one per class)
        # For binary classification, take the SHAP values for the positive class (index 1)
        if isinstance(shap_values, list):
            # Case 1: shap_values is a list (older SHAP versions or some models)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            # print(f"[DEBUG WeightedExplainer] After taking class 1 from list, shape: {shap_values.shape}")
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Case 2: shap_values is a 3D array with shape (n_samples, n_features, n_classes)
            # For binary classification, take the SHAP values for class 1 (positive class)
            shap_values = shap_values[:, :, 1]
            # print(f"[DEBUG WeightedExplainer] After taking class 1 from 3D array, shape: {shap_values.shape}")

        # Ensure shap_values is 1D by squeezing out all dimensions of size 1
        # This prevents issues with varphi having unexpected dimensions like (n_samples, 1, n_features)
        shap_values = np.squeeze(shap_values)
        # print(f"[DEBUG WeightedExplainer] After squeeze, shape: {shap_values.shape}")

        return shap_values


class JointProbabilityExplainer:
    """
    This class provides SHAP explanations for model predictions across multiple instances,
    using joint probability distributions to weight the baseline data for each instance.
    """

    def __init__(self, model, random_state=42, feature_names=None):
        """
        Initializes the JointProbabilityExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to compute SHAP values using weighted baseline data.
        :param random_state: Random seed for reproducible sampling. Default is 42.
        :param feature_names: List of feature names to use when creating DataFrames.
        """
        self.model = model
        self.weighted_explainer = WeightedExplainer(model, random_state=random_state, feature_names=feature_names)

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
        # print(f"[DEBUG JointProbabilityExplainer] Computing SHAP values for {len(X)} instances")
        results = []
        for i, (x, weights) in enumerate(zip(X, joint_probs)):
            result = self.weighted_explainer.explain_instance(
                x,
                X_baseline,
                weights,
                sample_size=sample_size,
                shap_sample_size=shap_sample_size,
            )
            # print(f"[DEBUG JointProbabilityExplainer] Instance {i}: returned shape {result.shape}")
            results.append(result)

        final_array = np.array(results)
        # print(f"[DEBUG JointProbabilityExplainer] Final np.array shape: {final_array.shape}")
        return final_array
