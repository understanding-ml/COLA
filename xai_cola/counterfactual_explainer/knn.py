import numpy as np
import pandas as pd
from xai_cola.data.base_data import BaseData
from xai_cola.ml_model.model import Model
from xai_cola.utils.logger_config import setup_logger
from .base_explainer import CounterFactualExplainer

FACTUAL_CLASS = 1
SHUFFLE_COUNTERFACTUAL = True

# 似乎是用来处理（非标准化后的）数据？
class KNN(CounterFactualExplainer):
    def __init__(self, ml_model:Model, data:BaseData=None):
        super().__init__(ml_model, data)
        """
        Initialize the KNN class
        
        Parameters:
        model: Pre-trained model
        data: use our wrapperred-data (NumpyData, PandasData, etc.)
        """ 
                        
    def generate_counterfactuals(
            self, 
            data:BaseData=None,
            n_neighbors:int=5,
            ) -> np.ndarray:
        """
        Generate counterfactuals for the given data
        
        Parameters:
        data: BaseData type, the factual data(don't need target column)
        params: parameters for specific counterfactual algorithm
        
        Returns:
        ndarray type counterfactual results
        """
        # Call the data processing logic from the parent class
        self._process_data(data)
        x_chosen = self.x_factual_pandas
        scaler = EfficientQuantileTransformer()
        scaler.fit(x_chosen)
        setup_logger()
        knn_explainer = KNNCounterfactuals(
            model=self.ml_model,
            X=x_chosen.values,
            n_neighbors=n_neighbors,
            distance="cityblock",
            scaler=scaler,
            max_samples=10000,
        )
        estimated = knn_explainer.get_multiple_counterfactuals(x_chosen.values)
        print(f'estimate:{estimated}')
        df_counterfactual = pd.DataFrame(
            np.array(estimated).reshape(x_chosen.shape[0] * n_neighbors, x_chosen.shape[1]),
            columns=x_chosen.columns,
        )
        if SHUFFLE_COUNTERFACTUAL:
            df_counterfactual = df_counterfactual.sample(frac=1).reset_index(drop=True)

        final_sample_num = min(x_chosen.shape[0], df_counterfactual.shape[0])
        X_factual = x_chosen.sample(final_sample_num).values
        X_counterfactual = df_counterfactual.sample(final_sample_num).values
        y_factual = self.ml_model.predict(X_factual)
        y_counterfactual = self.ml_model.predict(X_counterfactual)

        return X_factual, X_counterfactual





"""
    Author: Emanuele Albini

    This module contains base classes and interfaces (protocol in Python jargon).

    Note: this module contains classes that are more general than needed for this package.
    This is to allow for future integration in a more general XAI package.

    Most of the interfaces, base classes and methods are self-explanatory.

"""

from abc import ABC, abstractmethod
import warnings
from typing import Union, List
from pprint import pformat
import json

try:
    # In Python >= 3.8 this functionality is included in the standard library
    from typing import Protocol
    from typing import runtime_checkable
except (ImportError, ModuleNotFoundError):
    # Python < 3.8 - Backward Compatibility through package
    from typing_extensions import Protocol
    from typing_extensions import runtime_checkable

__all__ = [
    "Scaler",
    "Model",
    "ModelWithDecisionFunction",
    "XGBWrapping",
    "Explainer",
    "ExplainerSupportsDynamicBackground",
    "BaseExplainer",
    "BaseSupportsDynamicBackground",
    "BaseGroupExplainer",
    "BackgroundGenerator",
    "CounterfactualMethod",
    "MultipleCounterfactualMethod",
    "MultipleCounterfactualMethodSupportsWrapping",
    "MultipleCounterfactualMethodWrappable",
    "BaseCounterfactualMethod",
    "BaseMultipleCounterfactualMethod",
    "TrendEstimatorProtocol",
    "ListOf2DArrays",
    "CounterfactualEvaluationScorer",
    "BaseCounterfactualEvaluationScorer",
]
ListOf2DArrays = Union[List[np.ndarray], np.ndarray]

class attrdict(dict):
    """
    Attributes-dict bounded structure for paramenters
    -> When a dictionary key is set the corresponding attribute is set
    -> When an attribute is set the corresponding dictionary key is set

    Usage:

        # Create the object
        args = AttrDict()

        args.a = 1
        print(args.a) # 1
        print(args['a']) # 1

        args['b'] = 2
        print(args.b) # 2
        print(args['b']) # 2

    """

    def __init__(self, *args, **kwargs):
        super(attrdict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def repr(self):
        return dict(self)

    def __repr__(self):
        return pformat(self.repr())

    def __str__(self):
        return self.__repr__()

    def update_defaults(self, d: dict):
        for k, v in d.items():
            self.setdefault(k, v)

    def save_json(self, file_name):
        with open(file_name, "w") as fp:
            json.dump(self.repr(), fp)

    def copy(self):
        return type(self)(self)


def np_sample(
    a: Union[np.ndarray, int],
    n: Union[int, None],
    replace: bool = False,
    seed: Union[None, int] = None,
    random_state: Union[None, int] = None,
    safe: bool = False,
) -> np.ndarray:
    """Randomly sample on axis 0 of a NumPy array

    Args:
        a (Union[np.ndarray, int]): The array to be samples, or the integer (max) for an `range`
        n (int or None): Number of samples to be draw. If None, it sample all the samples.
        replace (bool, optional): Repeat samples or not. Defaults to False.
        seed (Union[None, int], optional): Random seed for NumPy. Defaults to None.
        random_state (Union[None, int], optional): Alias for seed. Defaults to None.
        safe (bool, optional) : Safely handle `n` or not. If True and replace = False, and n > len(a), then n = len(a)

    Returns:
        np.ndarray: A random sample
    """
    assert random_state is None or seed is None

    if random_state is not None:
        seed = random_state

    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random

    # Range case
    if isinstance(a, int):
        if safe and n > a:
            n = a
        return random_state.choice(a, n, replace=replace)
    # Array sampling case
    else:
        if n is None:
            n = len(a)
        if safe and n > len(a):
            n = len(a)
        return a[random_state.choice(a.shape[0], n, replace=replace)]


# ------------------- MODELs, etc. -------------------------


@runtime_checkable
class Model(Protocol):
    """Protocol for a ML model"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


@runtime_checkable
class ModelWithDecisionFunction(Model, Protocol):
    """Protocol for a Model with a decision function as well"""

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        pass


@runtime_checkable
class XGBWrapping(Model, Protocol):
    """Protocol for an XGBoost model wrapper"""

    def get_booster(self):
        pass


@runtime_checkable
class Scaler(Protocol):
    """Protocol for a Scaler"""

    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass


# ------------------- Explainers, etc. -------------------------


class BaseClass(ABC):
    """Base class for all explainability methods"""

    def __init__(
        self, model: Model, scaler: Union[Scaler, None] = None, random_state: int = 2021
    ):

        self._model = model
        self._scaler = scaler
        self.random_state = random_state

    # model and scaler cannot be changed at runtime. Set as properties.
    @property
    def model(self):
        return self._model

    @property
    def scaler(self):
        return self._scaler

    def preprocess(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise ValueError("Must pass a NumPy array.")

        if len(X.shape) != 2:
            raise ValueError("The input data must be a 2D matrix.")

        if X.shape[0] == 0:
            raise ValueError(
                "An empty array was passed! You must pass a non-empty array of samples in order to generate explanations."
            )

        return X

    def sample(self, X: np.ndarray, n: int):
        if n is not None:
            X = np_sample(X, n, random_state=self.random_state, safe=True)

        return X

    def scale(self, X: np.ndarray):
        if self.scaler is None:
            return X
        else:
            return self.scaler.transform(X)


@runtime_checkable
class Explainer(Protocol):
    """Protocol for an Explainer (a feature attribution/importance method).

    Attributes:
        model (Model): The model for which the feature importance is computed
        scaler (Scaler, optional): The scaler for the data. Default to None (i.e., no scaling).

    Methods:
        get_attributions(X): Returns the feature attributions.

    Optional Methods:
        get_trends(X): Returns the feature trends.
        get_backgrounds(X): Returns the background datasets.

    To build a new explainer one can easily extend BaseExplainer.
    """

    model: Model
    scaler: Union[Scaler, None]

    def get_attributions(self, X):
        pass

    # Optional
    # def get_trends(self, X):
    #     pass

    # def get_backgrounds(self, X):
    #     pass


@runtime_checkable
class SupportsDynamicBackground(Protocol):
    """Additional Protocol for a class that supports at-runtime change of the background data."""

    @property
    def data(self):
        pass

    @data.setter
    def data(self, data):
        pass


@runtime_checkable
class ExplainerSupportsDynamicBackground(
    Explainer, SupportsDynamicBackground, Protocol
):
    """Protocol for an Explainer that supports at-runtime change of the background data"""

    pass


class BaseExplainer(BaseClass, ABC):
    """Base class for a feature attribution/importance method"""

    @abstractmethod
    def get_attributions(self, X: np.ndarray) -> np.ndarray:
        """Generate the feature attributions for query instances X"""
        pass

    def get_trends(self, X: np.ndarray) -> np.ndarray:
        """Generate the feature trends for query instances X"""
        raise NotImplementedError("trends method is not implemented!")

    def get_backgrounds(self, X: np.ndarray) -> np.ndarray:
        """Returns the background datasets for query instances X"""
        raise NotImplementedError("get_backgrounds method is not implemented!")

    def __call__(self, X: np.ndarray) -> attrdict:
        """Returns the explanations

        Args:
            X (np.ndarray): The query instances

        Returns:
            attrdict: An attrdict (i.e., a dict which fields can be accessed also through attributes) with the following attributes:
            - .values : the feature attributions
            - .backgrounds : the background datasets (if any)
            - .trends : the feature trends (if any)
        """
        X = self.preprocess(X)
        return attrdict(
            values=self.get_attributions(X),
            backgrounds=self.get_backgrounds(X),
            trends=self.get_trends(X),
        )

    # Alias for 'get_attributions' for backward-compatibility
    def shap_values(self, *args, **kwargs):
        return self.get_attributions(*args, **kwargs)


class BaseSupportsDynamicBackground(ABC):
    """Base class for a class that supports at-runtime change of the background data."""

    @property
    def data(self):
        if self._data is None:
            self._raise_data_error()
        return self._data

    def _raise_data_error(self):
        raise ValueError("Must set background data first.")

    @data.setter
    @abstractmethod
    def data(self, data):
        pass


class BaseGroupExplainer:
    """Base class for an explainer (feature attribution) for groups of features."""

    def preprocess_groups(self, feature_groups: List[List[int]], nb_features):

        features_in_groups = sum(feature_groups, [])
        nb_groups = len(feature_groups)

        if nb_groups > nb_features:
            raise ValueError("There are more groups than features.")

        if len(set(features_in_groups)) != len(features_in_groups):
            raise ValueError("Some features are in multiple groups!")

        if len(set(features_in_groups)) < nb_features:
            raise ValueError("Not all the features are in groups")

        if any([len(x) == 0 for x in feature_groups]):
            raise ValueError("Some feature groups are empty!")

        return feature_groups


# ------------------------------------- BACKGROUND GENERATOR --------------------------------------


@runtime_checkable
class BackgroundGenerator(Protocol):
    """Protocol for a Background Generator: can be used together with an explainer to dynamicly generate backgrounds for each instance (see `composite`)"""

    def get_backgrounds(self, X: np.ndarray) -> ListOf2DArrays:
        """Returns the background datasets for the query instances.

        Args:
            X (np.ndarray): The query instances.

        Returns:
            ListOf2DArrays: The background datasets.
        """
        pass


class BaseBackgroundGenerator(BaseClass, ABC, BackgroundGenerator):
    """Base class for a background generator."""

    @abstractmethod
    def get_backgrounds(self, X: np.ndarray) -> ListOf2DArrays:
        pass


# ------------------------------------- TREND ESTIMATOR --------------------------------------


@runtime_checkable
class TrendEstimatorProtocol(Protocol):
    """Protocol for a feature Trend Estimator"""

    def predict(self, X: np.ndarray, YY: ListOf2DArrays) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass


# ------------------- Counterfactuals, etc. -------------------------


@runtime_checkable
class CounterfactualMethod(Protocol):
    """Protocol for a counterfactual generation method (that generate a single counterfactual per query instance)."""

    model: Model

    def get_counterfactuals(self, X: np.ndarray) -> np.ndarray:
        pass


@runtime_checkable
class MultipleCounterfactualMethod(CounterfactualMethod, Protocol):
    """Protocol for a counterfactual generation method (that generate a single OR MULTIPLE counterfactuals per query instance)."""

    def get_multiple_counterfactuals(self, X: np.ndarray) -> ListOf2DArrays:
        pass


class BaseCounterfactualMethod(BaseClass, ABC, CounterfactualMethod):
    """Base class for a counterfactual generation method (that generate a single counterfactual per query instance)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.invalid_counterfactual = "raise"

    def _invalid_response(self, invalid: Union[None, str]) -> str:
        invalid = invalid or self.invalid_counterfactual
        assert invalid in ("nan", "raise", "ignore")
        return invalid

    def postprocess(
        self,
        X: np.ndarray,
        XC: np.ndarray,
        invalid: Union[None, str] = None,
    ) -> np.ndarray:
        """Post-process counterfactuals

        Args:
            X (np.ndarray : nb_samples x nb_features): The query instances
            XC (np.ndarray : nb_samples x nb_features): The counterfactuals
            invalid (Union[None, str], optional): It can have the following values. Defaults to None ('raise').
            - 'nan': invalid counterfactuals (non changing prediction) will be marked with NaN
            - 'raise': an error will be raised if invalid counterfactuals are passed
            - 'ignore': Nothing will be node. Invalid counterfactuals will be returned.

        Returns:
            np.ndarray: The post-processed counterfactuals
        """

        invalid = self._invalid_response(invalid)

        # Mask with the non-flipped counterfactuals
        not_flipped_mask = self.model.predict(X) == self.model.predict(XC)
        if not_flipped_mask.sum() > 0:
            if invalid == "raise":
                self._raise_invalid()
            elif invalid == "nan":
                self._warn_invalid()
                XC[not_flipped_mask, :] = np.nan

        return XC

    def _warn_invalid(self):
        warnings.warn(
            "!! ERROR: Some counterfactuals are NOT VALID (will be set to NaN)"
        )

    def _raise_invalid(self):
        raise RuntimeError("Invalid counterfactuals")

    def _raise_nan(self):
        raise RuntimeError("NaN counterfactuals are generated before post-processing.")

    def _raise_inf(self):
        raise RuntimeError(
            "+/-inf counterfactuals are generated before post-processing."
        )

    @abstractmethod
    def get_counterfactuals(self, X: np.ndarray) -> np.ndarray:
        pass


class BaseMultipleCounterfactualMethod(BaseCounterfactualMethod):
    """Base class for a counterfactual generation method (that generate a single OR MULTIPLE counterfactuals per query instance)."""

    def multiple_postprocess(
        self,
        X: np.ndarray,
        XX_C: ListOf2DArrays,
        invalid: Union[None, str] = None,
        allow_nan: bool = True,
        allow_inf: bool = False,
    ) -> ListOf2DArrays:
        """Post-process multiple counterfactuals

        Args:
            X (np.ndarray : nb_samples x nb_features): The query instances
            XX_C (ListOf2DArrays : nb_samples x nb_counterfactuals x nb_features): The counterfactuals
            invalid (Union[None, str], optional): It can have the following values. Defaults to None ('raise').
            - 'nan': invalid counterfactuals (non changing prediction) will be marked with NaN
            - 'raise': an error will be raised if invalid counterfactuals are passed
            - 'ignore': Nothing will be node. Invalid counterfactuals will be returned.
            allow_nan (bool, optional): If True, allows NaN counterfactuals a input (invalid). If False, it raises an error. Defaults to True.
            allow_inf (bool, optional): If True, allows infinite in counterfactuals. If False, it raise an error. Defaults to False.

        Returns:
            ListOf2DArrays : The post-processed counterfactuals
        """

        invalid = self._invalid_response(invalid)

        # Reshape (for zero-length arrays)
        XX_C = [X_C.reshape(-1, X.shape[1]) for X_C in XX_C]

        # Check for NaN and Inf
        for XC in XX_C:
            if not allow_nan and np.isnan(XC).sum() != 0:
                self._raise_nan()
            if not allow_inf and np.isinf(XC).sum() != 0:
                self._raise_inf()

        # Mask with the non-flipped counterfactuals
        nb_counters = np.array([X_c.shape[0] for X_c in XX_C])
        not_flipped_mask = np.equal(
            np.repeat(self.model.predict(X), nb_counters),
            self.model.predict(np.concatenate(XX_C, axis=0)),
        )
        if not_flipped_mask.sum() > 0:
            if invalid == "raise":
                print("X, f(X) :", X, self.model.predict(X))
                print(
                    "X_C, f(X_C) :",
                    XX_C,
                    self.model.predict(np.concatenate(XX_C, axis=0)),
                )
                self._raise_invalid()
            elif invalid == "nan":
                self._warn_invalid()
                sections = np.cumsum(nb_counters[:-1])
                not_flipped_mask = np.split(
                    not_flipped_mask, indices_or_sections=sections
                )

                # Set them to nan
                for i, nfm in enumerate(not_flipped_mask):
                    XX_C[i][nfm, :] = np.nan

        return XX_C

    def multiple_trace_postprocess(self, X, XTX_counter, invalid=None):
        invalid = self._invalid_response(invalid)

        # Reshape (for zero-length arrays)
        XTX_counter = [
            [X_C.reshape(-1, X.shape[1]) for X_C in TX_C] for TX_C in XTX_counter
        ]

        # Mask with the non-flipped counterfactuals
        shapess = [[X_C.shape[0] for X_C in TX_C] for TX_C in XTX_counter]
        shapes = [sum(S) for S in shapess]

        X_counter = np.concatenate(
            [np.concatenate(TX_C, axis=0) for TX_C in XTX_counter], axis=0
        )
        not_flipped_mask = np.equal(
            np.repeat(self.model.predict(X), shapes),
            self.model.predict(X_counter),
        )
        if not_flipped_mask.sum() > 0:
            if invalid == "raise":
                self._raise_invalid()
            elif invalid == "nan":
                self._warn_invalid()
                sections = np.cumsum(shapes[:-1])
                sectionss = [np.cumsum(s[:-1]) for s in shapess]
                not_flipped_mask = np.split(
                    not_flipped_mask, indices_or_sections=sections
                )
                not_flipped_mask = [
                    np.split(NFM, indices_or_sections=s)
                    for NFM, s in zip(not_flipped_mask, sectionss)
                ]

                # Set them to nan
                for i, NFM in enumerate(not_flipped_mask):
                    for j, nfm in enumerate(NFM):
                        X_counter[i][j][nfm, :] = np.nan

        return XTX_counter

    @abstractmethod
    def get_multiple_counterfactuals(self, X: np.ndarray) -> ListOf2DArrays:
        pass

    def get_counterfactuals(self, X: np.ndarray) -> np.ndarray:
        return np.array([X_C[0] for X_C in self.get_multiple_counterfactuals(X)])

    # Alias backward compatibility
    def diverse_postprocess(self, *args, **kwargs):
        return self.multiple_postprocess(*args, **kwargs)

    def diverse_trace_postprocess(self, *args, **kwargs):
        return self.multiple_trace_postprocess(*args, **kwargs)


@runtime_checkable
class Wrappable(Protocol):
    verbose: Union[int, bool]


@runtime_checkable
class SupportsWrapping(Protocol):
    @property
    def data(self):
        pass

    @data.setter
    @abstractmethod
    def data(self, data):
        pass


@runtime_checkable
class MultipleCounterfactualMethodSupportsWrapping(
    MultipleCounterfactualMethod, SupportsWrapping, Protocol
):
    """Protocol for a counterfactual method that can be wrapped by another one
    (i.e., the output of a SupportsWrapping method can be used as background data of another)
    """

    pass


@runtime_checkable
class MultipleCounterfactualMethodWrappable(
    MultipleCounterfactualMethod, Wrappable, Protocol
):
    """Protocol for a counterfactual method that can used as wrapping for another one
    (i.e., a Wrappable method can use the ouput of an another CFX method as input)"""

    pass


# ------------------------ EVALUATION -----------------------


@runtime_checkable
class CounterfactualEvaluationScorer(Protocol):
    """Protocol for an evaluation method that returns an array of scores (float) for a list of counterfactuals."""

    def score(self, X: np.ndarray) -> np.ndarray:
        pass


class BaseCounterfactualEvaluationScorer(ABC):
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        pass




"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------BELOW IS THE INITIAL IMPLEMENTATION OF KNN COUNTERFACTUALS------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
"""
    Author: Emanuele Albini

    Implementation of K-Nearest Neighbours Counterfactuals
"""

__all__ = ["KNNCounterfactuals"]

from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
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


"""
----------------------------------------------------------------------------------cutting---------------------------------------------------------------------------------------------------------
"""
