import shap
from ..explainers import pshap
import ot
import numpy as np

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"


def COLA(X_factual, varphi, q, C, replace):
    action_indice = np.random.choice(
        a=varphi.size,
        size=C,
        p=varphi.flatten(),
        replace=replace,
    )
    action_indice = np.unique(action_indice)

    # Convert flat indices back to 2D indices
    i_indice, k_indice = np.unravel_index(action_indice, varphi.shape)

    Z_counterfactual = X_factual.copy()
    if C > 0:
        # Set values at selected 2D indices
        values_from_q = q[i_indice, k_indice]
        Z_counterfactual[i_indice, k_indice] = values_from_q

    return Z_counterfactual, action_indice


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


class Policy:
    def __init__(self, model, X_factual, X_counterfactual):
        self.model = model
        self.X_factual = X_factual
        self.X_counterfactual = X_counterfactual
        self.N, self.M = self.X_factual.shape[0], self.X_counterfactual.shape[0]


class CounterfactualPolicy(Policy):
    def __init__(self, model, X_factual, X_counterfactual):
        super().__init__(model, X_factual, X_counterfactual)


class TrainsetPolicy(Policy):
    def __init__(self, model, X_factual, X_train, X_counterfactual):
        super().__init__(model, X_factual, X_counterfactual)
        self.X_train_sampled = X_train.sample(self.M).values


class TrainUniformDistributionPolicy(TrainsetPolicy):
    def __init__(self, model, X_factual, X_train, X_counterfactual, method="avg"):
        super().__init__(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
        )
        self.method = method

    def compute_policy(self):
        p = get_uniform_distribution_matrix(self.N, self.M)

        shap_values = shap.KernelExplainer(
            self.model.predict_proba, self.X_train_sampled
        ).shap_values(self.X_factual, nsamples=SHAP_SAMPLE_SIZE)

        varphi = convert_matrix_to_policy(shap_values)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class TrainOptimalTransportPolicy(TrainsetPolicy):
    def __init__(
        self, model, X_factual, X_train, X_counterfactual, reg=0, method="avg"
    ):
        super().__init__(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
        )
        self.reg = reg
        self.method = method
        self.ot_cost = ot.dist(self.X_factual, self.X_counterfactual, p=2)

    def compute_policy(self):
        if self.reg <= 0:
            probs_matrix = ot.emd(
                np.ones(self.N) / self.N, np.ones(self.M) / self.M, self.ot_cost
            )
        else:
            probs_matrix = ot.bregman.sinkhorn(
                np.ones(self.N) / self.N,
                np.ones(self.M) / self.M,
                self.ot_cost,
                reg=self.reg,
                numItermax=5000,
            )
        p = convert_matrix_to_policy(probs_matrix)

        shap_values = shap.KernelExplainer(
            self.model.predict_proba, self.X_train_sampled
        ).shap_values(self.X_factual, nsamples=SHAP_SAMPLE_SIZE)

        varphi = convert_matrix_to_policy(shap_values)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualUniformDistributionPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_policy(self):
        p = get_uniform_distribution_matrix(self.N, self.M)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy(shap_values)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualSingleMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_policy(self):
        p = get_one_one_distribution_matrix(self.N, self.M)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy(shap_values)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualRandomMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_policy(self):
        p = get_random_distribution_matrix(self.N, self.M)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy(shap_values)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualExactMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method
        assert self.N == self.M

    def compute_policy(self):
        p = get_exact_one_one_matrix(self.N)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy(shap_values)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualOptimalTransportPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, reg=0, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.reg = reg
        self.method = method
        self.ot_cost = ot.dist(self.X_factual, self.X_counterfactual, p=2)

    def compute_policy(self):
        if self.reg <= 0:
            probs_matrix = ot.emd(
                np.ones(self.N) / self.N, np.ones(self.M) / self.M, self.ot_cost
            )
        else:
            probs_matrix = ot.bregman.sinkhorn(
                np.ones(self.N) / self.N,
                np.ones(self.M) / self.M,
                self.ot_cost,
                reg=self.reg,
                numItermax=5000,
            )
        p = convert_matrix_to_policy(probs_matrix)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy(shap_values)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


def compute_intervention_policy(
    model,
    X_train,
    X_factual,
    X_counterfactual,
    shapley_method,
    Avalues_method,
):
    if shapley_method == "CF_UniformMatch":
        return CounterfactualUniformDistributionPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "Train_Distri":
        return TrainUniformDistributionPolicy(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "Train_OTMatch":
        return TrainOptimalTransportPolicy(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_ExactMatch":
        return CounterfactualExactMatchingPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_RandomMatch":
        return CounterfactualRandomMatchingPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    else:
        shapley_method_string_list = shapley_method.split("_")
        entropic_ot = can_convert_to_float(shapley_method_string_list[-1])
        if entropic_ot:
            reg = float(shapley_method_string_list[-1])
            shapley_method = "_".join(shapley_method_string_list[:-1])
        else:
            reg = 0
        if shapley_method == "CF_OTMatch":
            return CounterfactualOptimalTransportPolicy(                        # use OT to calculate the joint distribution 
                model=model,
                X_factual=X_factual,
                X_counterfactual=X_counterfactual,
                reg=reg,
                method=Avalues_method,
            ).compute_policy()
        else:
            raise NotImplementedError


def can_convert_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_uniform_distribution_matrix(N, M):
    uniform_matrix = np.full((N, M), fill_value=1)
    return convert_matrix_to_policy(uniform_matrix)


def get_exact_one_one_matrix(N):
    eye_matrix = np.eye(N) / N
    return convert_matrix_to_policy(eye_matrix)


def get_one_one_distribution_matrix(N, M):
    probs_matrix = np.zeros((N, M))
    for i in range(N):
        probs_matrix[i, np.random.randint(0, M)] = 1 / N
    return convert_matrix_to_policy(probs_matrix)


def get_random_distribution_matrix(N, M):
    probs_matrix = np.random.rand(N, M)
    return convert_matrix_to_policy(probs_matrix)


def convert_matrix_to_policy(matrix):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P
