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

import numpy as np


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
