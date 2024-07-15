"""
    Author: Emanuele Albini

    Implementation of K-Nearest Neighbours Counterfactuals
"""

__all__ = ["KNNCounterfactuals"]

from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


from explainers.base import (
    BaseMultipleCounterfactualMethod,
    Model,
    Scaler,
    ListOf2DArrays,
)

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_array


class keydefaultdict(defaultdict):
    """
    Extension of defaultdict that support
    passing the key to the default_factory
    """

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key, "Must pass a default factory with a single argument.")
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class KNNCounterfactuals(BaseMultipleCounterfactualMethod):
    """Returns the K Nearest Neighbours of the query instance with a different prediction."""

    def __init__(
        self,
        model: Model,
        scaler: Union[None, Scaler],
        X: np.ndarray,
        nb_diverse_counterfactuals: Union[None, int, float] = None,
        n_neighbors: Union[None, int, float] = None,
        distance: str = None,
        max_samples: int = int(1e10),
        random_state: int = 2021,
        verbose: int = 0,
        **distance_params,
    ):
        """

        Args:
            model (Model): The model.
            scaler (Union[None, Scaler]): The scaler for the data.
            X (np.ndarray): The background dataset.
            nb_diverse_counterfactuals (Union[None, int, float], optional): Number of counterfactuals to generate. Defaults to None.
            n_neighbors (Union[None, int, float], optional): Number of neighbours to generate. Defaults to None.
                Note that this is an alias for nb_diverse_counterfactuals in this class.
            distance (str, optional): The distance metric to use for K-NN. Defaults to None.
            max_samples (int, optional): Number of samples of the background to use at most. Defaults to int(1e10).
            random_state (int, optional): Random seed. Defaults to 2021.
            verbose (int, optional): Level of verbosity. Defaults to 0.
            **distance_params: Additional parameters for the distance metric
        """

        assert (
            nb_diverse_counterfactuals is not None or n_neighbors is not None
        ), "nb_diverse_counterfactuals or n_neighbors must be set."

        super().__init__(model, scaler, random_state)

        self._metric, self._metric_params = distance, distance_params
        self.__nb_diverse_counterfactuals = nb_diverse_counterfactuals
        self.__n_neighbors = n_neighbors
        self.max_samples = max_samples

        self.data = X
        self.verbose = verbose

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._raw_data = self.preprocess(data)
        if self.max_samples < len(self._raw_data):
            self._raw_data = self.sample(self._raw_data, n=self.max_samples)
        self._preds = self.model.predict(self._raw_data)

        # In the base class this two information are equivalent
        if self.__n_neighbors is None:
            self.__n_neighbors = self.__nb_diverse_counterfactuals
        if self.__nb_diverse_counterfactuals is None:
            self.__nb_diverse_counterfactuals = self.__n_neighbors

        def get_nb_of_items(nb):
            if np.isinf(nb):
                return keydefaultdict(lambda pred: self._data[pred].shape[0])
            elif isinstance(nb, int) and nb >= 1:
                return keydefaultdict(lambda pred: min(nb, self._data[pred].shape[0]))
            elif isinstance(nb, float) and nb <= 1.0 and nb > 0.0:
                return keydefaultdict(
                    lambda pred: int(max(1, round(len(self._data[pred]) * nb)))
                )
            else:
                raise ValueError(
                    "Invalid n_neighbors, it must be the number of neighbors (int) or the fraction of the dataset (float)"
                )

        self._n_neighbors = get_nb_of_items(self.__n_neighbors)
        self._nb_diverse_counterfactuals = get_nb_of_items(
            self.__nb_diverse_counterfactuals
        )

        # We will be searching neighbors of a different class
        self._data = keydefaultdict(lambda pred: self._raw_data[self._preds != pred])

        self._nn = keydefaultdict(
            lambda pred: NearestNeighbors(
                n_neighbors=self._n_neighbors[pred],
                metric=self._metric,
                p=self._metric_params["p"] if "p" in self._metric_params else 2,
                metric_params=self._metric_params,
            ).fit(self.scale(self._data[pred]))
        )

    def get_counterfactuals(self, X: np.ndarray) -> np.ndarray:
        """Generate the closest counterfactual for each query instance"""

        # Pre-process
        X = self.preprocess(X)

        preds = self.model.predict(X)
        preds_indices = {
            pred: np.argwhere(preds == pred).flatten() for pred in np.unique(preds)
        }

        X_counter = np.zeros_like(X)

        for pred, indices in preds_indices.items():
            _, neighbors_indices = self._nn[pred].kneighbors(
                self.scale(X), n_neighbors=1
            )
            X_counter[indices] = self._data[pred][neighbors_indices.flatten()]

        # Post-process
        X_counter = self.postprocess(X, X_counter, invalid="raise")

        return X_counter

    def get_multiple_counterfactuals(self, X: np.ndarray) -> ListOf2DArrays:
        """Generate the multiple closest counterfactuals for each query instance"""

        # Pre-condition
        assert self.__n_neighbors == self.__nb_diverse_counterfactuals, (
            "n_neighbors and nb_diverse_counterfactuals are set to different values"
            f"({self.__n_neighbors} != {self.__nb_diverse_counterfactuals})."
            "When both are set they must be set to the same value."
        )

        # Pre-process
        X = self.preprocess(X)

        preds = self.model.predict(X)
        preds_indices = {
            pred: np.argwhere(preds == pred).flatten() for pred in np.unique(preds)
        }

        X_counter = [
            np.full((self._nb_diverse_counterfactuals[preds[i]], X.shape[1]), np.nan)
            for i in range(X.shape[0])
        ]

        for pred, indices in preds_indices.items():
            _, neighbors_indices = self._nn[pred].kneighbors(
                self.scale(X[indices]), n_neighbors=None
            )
            counters = self._data[pred][neighbors_indices.flatten()].reshape(
                len(indices), self._nb_diverse_counterfactuals[pred], -1
            )
            for e, i in enumerate(indices):
                # We use :counters[e].shape[0] so it raises an exception if shape are not coherent.
                X_counter[i][: counters[e].shape[0]] = counters[e]

        # Post-process
        X_counter = self.diverse_postprocess(X, X_counter, invalid="raise")

        return X_counter


class EfficientQuantileTransformer(QuantileTransformer):
    """
    This class directly extends and improve the efficiency of scikit-learn QuantileTransformer

    Note: The efficient implementation will be only used if:
    - The input are NumPy arrays (NOT scipy sparse matrices)
    The flag self.smart_fit_ marks when the efficient implementation is being used.

    """

    def __init__(
        self,
        *,
        subsample=np.inf,
        random_state=None,
        copy=True,
        overflow=None,  # "nan" or "sum"
    ):
        """Initialize the transformer

        Args:
            subsample (int, optional): Number of samples to use to create the quantile space. Defaults to np.inf.
            random_state (int, optional): Random seed (sampling happen only if subsample < number of samples fitted). Defaults to None.
            copy (bool, optional): If False, passed arrays will be edited in place. Defaults to True.
            overflow (str, optional): Overflow strategy. Defaults to None.
            When doing the inverse transformation if a quantile > 1 or < 0 is passed then:
                - None > Nothing is done. max(0, min(1, q)) will be used. The 0% or 100% reference will be returned.
                - 'sum' > It will add proportionally, e.g., q = 1.2 will result in adding 20% more quantile to the 100% reference.
                - 'nan' > It will return NaN
        """
        self.ignore_implicit_zeros = False
        self.n_quantiles_ = np.inf
        self.output_distribution = "uniform"
        self.subsample = subsample
        self.random_state = random_state
        self.overflow = overflow
        self.copy = copy

    def _smart_fit(self, X, random_state):
        n_samples, n_features = X.shape
        self.references_ = []
        self.quantiles_ = []
        for col in X.T:
            # Do sampling if necessary
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample, replace=False
                )
                col = col.take(subsample_idx, mode="clip")
            col = np.sort(col)
            quantiles = np.sort(np.unique(col))
            references = (
                0.5
                * np.array(
                    [
                        np.searchsorted(col, v, side="left")
                        + np.searchsorted(col, v, side="right")
                        for v in quantiles
                    ]
                )
                / n_samples
            )
            self.quantiles_.append(quantiles)
            self.references_.append(references)

    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        y : None
            Ignored.
        Returns
        -------
        self : object
           Fitted transformer.
        """

        if self.subsample <= 1:
            raise ValueError(
                "Invalid value for 'subsample': %d. "
                "The number of subsamples must be at least two." % self.subsample
            )

        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples = X.shape[0]

        if n_samples <= 1:
            raise ValueError(
                "Invalid value for samples: %d. "
                "The number of samples to fit for must be at least two." % n_samples
            )

        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.smart_fit_ = not sparse.issparse(X)
        if self.smart_fit_:  # <<<<<- New case
            self._smart_fit(X, rng)
        else:
            raise NotImplementedError(
                "EfficientQuantileTransformer handles only NON-sparse matrices!"
            )

        return self

    def _smart_transform_col(self, X_col, quantiles, references, inverse):
        """Private function to transform a single feature."""

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        # Simply Interpolate
        if not inverse:
            X_col[isfinite_mask] = np.interp(X_col_finite, quantiles, references)
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, references, quantiles)

        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        try:
            X = self._validate_data(
                X,
                reset=in_fit,
                accept_sparse=False,
                copy=copy,
                dtype=FLOAT_DTYPES,
                force_all_finite="allow-nan",
            )
        except AttributeError:  # Old sklearn version (_validate_data do not exists)
            X = check_array(
                X,
                accept_sparse=False,
                copy=self.copy,
                dtype=FLOAT_DTYPES,
                force_all_finite="allow-nan",
            )

        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if (
                not accept_sparse_negative
                and not self.ignore_implicit_zeros
                and (sparse.issparse(X) and np.any(X.data < 0))
            ):
                raise ValueError(
                    "QuantileTransformer only accepts" " non-negative sparse matrices."
                )

        # check the output distribution
        if self.output_distribution not in ("normal", "uniform"):
            raise ValueError(
                "'output_distribution' has to be either 'normal'"
                " or 'uniform'. Got '{}' instead.".format(self.output_distribution)
            )

        return X

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.
        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self._smart_transform_col(
                X[:, feature_idx],
                self.quantiles_[feature_idx],
                self.references_[feature_idx],
                inverse,
            )

        return X

    def transform(self, X):
        """Feature-wise transformation of the data.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self, ["quantiles_", "references_", "smart_fit_"])
        X = self._check_inputs(X, in_fit=False, copy=self.copy)
        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self, ["quantiles_", "references_", "smart_fit_"])
        X = self._check_inputs(
            X, in_fit=False, accept_sparse_negative=False, copy=self.copy
        )

        if self.overflow is None:
            T = self._transform(X, inverse=True)
        elif self.overflow == "nan":
            NaN_mask = np.ones(X.shape)
            NaN_mask[(X > 1) | (X < 0)] = np.nan
            T = NaN_mask * self._transform(X, inverse=True)

        elif self.overflow == "sum":
            ones = self._transform(np.ones(X.shape), inverse=True)
            zeros = self._transform(np.zeros(X.shape), inverse=True)

            # Standard computation
            T = self._transform(X.copy(), inverse=True)

            # Deduct already computed part
            X = np.where((X > 0), np.maximum(X - 1, 0), X)

            # After this X > 0 => Remaining quantile > 1.00
            # and X < 0 => Remaining quantile < 0.00

            T = T + (X > 1) * np.floor(X) * (ones - zeros)
            X = np.where((X > 1), np.maximum(X - np.floor(X), 0), X)
            T = T + (X > 0) * (ones - self._transform(1 - X.copy(), inverse=True))

            T = T - (X < -1) * np.floor(-X) * (ones - zeros)
            X = np.where((X < -1), np.minimum(X + np.floor(-X), 0), X)
            T = T - (X < 0) * (self._transform(-X.copy(), inverse=True) - zeros)

            # Set the NaN the values that have not been reached after a certaing amount of iterations
            # T[(X > 0) | (X < 0)] = np.nan

        else:
            raise ValueError("Invalid value for overflow.")

        return T
