"""
    The majority of code originates from https://github.com/youlei202/distributional-counterfactual-explanation
   
    Paper for reference:
    L. You, L. Cao, M. Nilsson, B. Zhao, and L. Lei.
    DIStributional COUNTerfactual Explanation With Optimal Transport

"""
import ot
import numpy as np
import pandas as pd
import math
from typing import Optional
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from xai_cola.ce_sparsifier.data import COLAData
from xai_cola.ce_sparsifier.models import Model
from xai_cola.ce_sparsifier.utils.logger_config import setup_logger
from xai_cola.ce_generator import auxiliary as aux
from .base_explainer import CounterFactualExplainer


SHUFFLE_COUNTERFACTUAL = False
logger = setup_logger()

"""
ATTENTION: The class DisCount is only compatible with PyTorch models (backend:"pytorch")
"""

class DisCount(CounterFactualExplainer):
    def __init__(self, ml_model: Model):
        """
        Initialize DisCount explainer

        Parameters:
        -----------
        ml_model : Model
            Pre-trained model wrapped in Model interface
            Must be a PyTorch model (backend='pytorch')
        """
        super().__init__(ml_model)

        # Verify that model is PyTorch
        if self.ml_model.backend != 'pytorch':
            raise ValueError(
                f"DisCount requires PyTorch models (backend='pytorch'), "
                f"but got backend='{self.ml_model.backend}'"
            )


    """
    Since we no more use the sample_num right now, we can remove the method 'get_factual_indices()'
    We will generate all the input data as factuals and return the counterfactuals throught the generate_counterfactuals method
    """
    # def get_factual_indices(self):  
    #     # input: X_factual_pandas(dataframe, without target_column), model, target_name, sample_num 
    #     X_test_ext = self.x_factual_pandas.copy()
    #     X_test_ext[self.target_name] = self.ml_model.predict(self.x_factual_pandas.values)
    #     sampling_weights = np.exp(X_test_ext[self.target_name].values.clip(min=0) * 4)
    #     indices = (X_test_ext.sample(self.sample_num, weights=sampling_weights)).index
    #     # Select indices from the target_name column according to weights, where entries with risk=1 are more likely to be selected
    #     return indices


    def generate_counterfactuals(
            self,
            data: COLAData=None,
            factual_class: int=1,
            lr: float=5e-2,
            n_proj: int=10,
            delta: float=0.15,
            U_1: float=0.4,
            U_2: float=0.02,
            l: float=0.15,
            r: float=1,
            max_iter: int=100,
            tau: float=1e1,
            silent: bool=False,
            explain_columns: list=None,
            ) -> tuple:
        """
        Implement the DisCount algorithm to generate counterfactuals.

        Parameters:
        -----------
        data : COLAData, optional
            Factual data wrapper
        factual_class : int, default=1
            The class of the factual data (Normally, we set the factual_class as 1
            as the prediction of factual data is 1. And we hope the prediction of counterfactual data is 0)
        lr : float, default=5e-2
            Learning rate
        n_proj : int, default=10
            Number of projections
        delta : float, default=0.15
            Trimming constant
        U_1 : float, default=0.4
            Upper bound for the Wasserstein distance
        U_2 : float, default=0.02
            Upper bound for the sliced Wasserstein distance
        l : float, default=0.15
            Lower bound for the interval narrowing
        r : float, default=1
            Upper bound for the interval narrowing
        max_iter : int, default=100
            Maximum number of iterations
        tau : float, default=1e1
            Step size (can't be too large or too small)
        silent : bool, default=False
            Whether to print the log information
        explain_columns : list, optional
            List of column names to use for explanation
            If None, use all columns from transformed data

        Returns:
        --------
        tuple: (factual_df, counterfactual_df)
            factual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
            counterfactual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Target column values for counterfactual are set to (1 - factual_class)

        Notes:
        ------
        - Only compatible with PyTorch models (backend='pytorch')
        - Supports data transformation via COLAData.transform_method parameter
        - Automatically handles inverse transformation of counterfactuals
        """
        self.lr = lr
        self.n_proj = n_proj
        self.delta = delta
        self.U_1 = U_1
        self.U_2 = U_2
        self.l = l
        self.r = r
        self.max_iter = max_iter
        self.tau = tau
        self.silent = silent

        # Call the data processing logic from the parent class
        self._process_data(data)

        x_chosen = self.x_factual_pandas  # factual, dataframe type, without target_column

        # DEBUG: Print original factual data
        if not silent:
            print("\n" + "="*80)
            print("DEBUG: Factual data (original space)")
            print("="*80)
            print(f"Shape: {x_chosen.shape}")
            print(f"Columns: {x_chosen.columns.tolist()}")
            print(x_chosen)
            print()

        # Apply transformation if needed
        if self.data.transform_method is not None:
            x_chosen_transformed = self.data._transform(x_chosen)

            # DEBUG: Print transformed factual data
            if not silent:
                print("\n" + "="*80)
                print("DEBUG: Factual data AFTER transform (transformed space)")
                print("="*80)
                print(f"Shape: {x_chosen_transformed.shape}")
                print(f"Columns: {x_chosen_transformed.columns.tolist()}")
                print(x_chosen_transformed)
                print()
        else:
            x_chosen_transformed = x_chosen

        # Determine explain_columns: use provided or default to all transformed columns
        if explain_columns is None:
            explain_columns = x_chosen_transformed.columns.tolist()

        df_factual_ext = x_chosen_transformed.copy()

        # Get predictions on transformed data if needed
        if self.data.transform_method is not None:
            pred_values = self.ml_model.predict(x_chosen_transformed.values)
        else:
            pred_values = self.ml_model.predict(x_chosen.values)

        df_factual_ext[self.target_name] = pred_values

        y_target = torch.FloatTensor(
            [1 - factual_class for _ in range(x_chosen_transformed.shape[0])]
        )

        # Get PyTorch model from wrapper
        pytorch_model = self.ml_model._get_wrapped_model().model

        discount_explainer = DistributionalCounterfactualExplainer(
            model = pytorch_model,
            df_X = x_chosen_transformed,
            explain_columns = explain_columns,
            y_target = y_target,
            lr = self.lr,
            n_proj = self.n_proj,
            delta = self.delta,
        )

        discount_explainer.optimize(
            U_1 = self.U_1,
            U_2 = self.U_2,
            l = self.l,
            r = self.r,
            max_iter = self.max_iter,
            tau = self.tau,
            silent = self.silent
        )

        # Get the complete optimized data (all features) from DisCount
        # Note: discount_explainer.best_X contains ALL features, not just explain_columns
        best_X_all_features = discount_explainer.best_X.detach().numpy()

        # Create DataFrame with all features
        df_counterfactual_transformed = pd.DataFrame(
            best_X_all_features,
            columns=x_chosen_transformed.columns,  # Use all original column names
            index=x_chosen_transformed.index,
        )

        # CRITICAL: Verify that the generated counterfactuals achieve the target prediction
        # This is the ONLY place we need to check predictions (in transformed space)
        if self.data.transform_method is not None:
            counterfactual_predictions = self.ml_model.predict(df_counterfactual_transformed.values)
        else:
            counterfactual_predictions = self.ml_model.predict(best_X_all_features)

        target_class = 1 - factual_class
        n_correct = np.sum(counterfactual_predictions == target_class)
        n_total = len(counterfactual_predictions)

        if not silent:
            print("\n" + "="*80)
            print("DEBUG: Counterfactual Prediction Verification (transformed space)")
            print("="*80)
            print(f"Target class: {target_class}")
            print(f"Counterfactual predictions: {counterfactual_predictions}")
            print(f"Successfully achieved target: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
            if n_correct < n_total:
                print(f"WARNING: {n_total - n_correct} samples did not achieve target prediction!")
            print()
            print("Note: These predictions are verified in transformed space (where model was trained)")
            print()

        # DEBUG: Print transformed data before inverse transform
        if not silent:
            print("\n" + "="*80)
            print("DEBUG: Counterfactual BEFORE inverse_transform (transformed space)")
            print("="*80)
            print(f"Shape: {df_counterfactual_transformed.shape}")
            print(f"Columns: {df_counterfactual_transformed.columns.tolist()}")
            print(df_counterfactual_transformed)
            print()

        # Inverse transform if needed
        if self.data.transform_method is not None:
            df_counterfactual = self.data._inverse_transform(df_counterfactual_transformed)

            # DEBUG: Print data after inverse transform
            if not silent:
                print("\n" + "="*80)
                print("DEBUG: Counterfactual AFTER inverse_transform (original space)")
                print("="*80)
                print(f"Shape: {df_counterfactual.shape}")
                print(f"Columns: {df_counterfactual.columns.tolist()}")
                print(df_counterfactual)
                print()
        else:
            df_counterfactual = df_counterfactual_transformed

        if SHUFFLE_COUNTERFACTUAL:
            df_counterfactual = df_counterfactual.sample(frac=1).reset_index(drop=True)

        # Prepare factual with target column - get directly from COLAData
        factual_df = self.data.get_factual_all()

        # Convert numerical features to int
        if self.data.numerical_features is not None:
            for num_feature in self.data.numerical_features:
                if num_feature in df_counterfactual.columns:
                    # First convert to numeric (handles object dtype from inverse_transform)
                    df_counterfactual[num_feature] = pd.to_numeric(df_counterfactual[num_feature], errors='coerce')
                    df_counterfactual[num_feature] = df_counterfactual[num_feature].round().astype(int)

        # Prepare counterfactual with target column
        # Use the predictions from transformed space (counterfactual_predictions calculated earlier)
        counterfactual_df = df_counterfactual.copy()
        counterfactual_df[self.target_name] = counterfactual_predictions

        # Ensure column order matches factual (target column at the end)
        all_columns = factual_df.columns.tolist()
        counterfactual_df = counterfactual_df[all_columns]

        # SUMMARY: Report final counterfactual predictions
        if not silent:
            print("\n" + "="*80)
            print("SUMMARY: Final Counterfactual DataFrame")
            print("="*80)
            print(f"Target class (desired): {target_class}")
            print(f"Counterfactual Risk column (model predictions): {counterfactual_predictions}")
            n_match = np.sum(counterfactual_predictions == target_class)
            print(f"Samples achieving target: {n_match}/{len(counterfactual_predictions)} ({100*n_match/len(counterfactual_predictions):.1f}%)")
            print("="*80)
            print()

        return factual_df, counterfactual_df


class DistributionalCounterfactualExplainer:
    def __init__(
        self,
        model,
        df_X,
        explain_columns,
        y_target,
        lr=0.1,
        init_eta=0.5,
        n_proj=50,
        delta=0.1,
        costs_vector=None,
    ):
        self.X = df_X.values
        # Find indices of explain_columns in df_X
        self.explain_indices = [df_X.columns.get_loc(col) for col in explain_columns]

        self.explain_columns = explain_columns

        # Set the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.from_numpy(self.X).float().to(self.device)

        # Move model to the appropriate device
        self.model = model.to(self.device)

        # Transfer data to the device
        self.X_prime = self.X.clone()

        noise = torch.randn_like(self.X_prime[:, self.explain_indices]) * 0.01
        self.X[:, self.explain_indices] = (
            self.X_prime[:, self.explain_indices] + noise
        ).to(self.device)

        self.X.requires_grad_(True).retain_grad()
        self.best_X = None
        self.Qx_grads = None
        self.optimizer = optim.SGD([self.X], lr=lr)

        self.y = self.model(self.X)
        self.y_prime = y_target.clone().to(self.device)
        self.best_y = None

        self.swd = SlicedWassersteinDivergence(
            self.X_prime[:, self.explain_indices].shape[1], n_proj=n_proj
        )
        self.wd = WassersteinDivergence()

        self.Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.best_gap = np.inf

        self.init_eta = torch.tensor(init_eta, dtype=torch.float, device=self.device)

        self.delta = delta
        self.found_feasible_solution = False

        if costs_vector is None:
            self.costs_vector = torch.ones(len(self.explain_indices)).float()
        else:
            self.costs_vector = torch.tensor(costs_vector).float()

        self.costs_vector_reshaped = self.costs_vector.reshape(1, -1)

        self.wd_list = []
        self.wd_upper_list = []
        self.wd_lower_list = []

        self.swd_list = []
        self.swd_upper_list = []
        self.swd_lower_list = []

        self.eta_list = []
        self.interval_left_list = []
        self.interval_right_list = []

    def _update_Q(self, mu_list, nu, eta):
        n, m = (
            self.X[:, self.explain_indices].shape[0],
            self.X_prime[:, self.explain_indices].shape[0],
        )

        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]

        # Compute the first term
        self.term1 = torch.tensor(0.0, dtype=torch.float).to(self.device)
        for k, theta in enumerate(thetas):
            mu = mu_list[k]
            mu = mu.to(self.device)
            for i in range(n):
                for j in range(m):
                    # Apply the costs to the features of X and X_prime
                    weighted_X = (
                        self.X[:, self.explain_indices] * self.costs_vector_reshaped
                    )
                    weighted_X_prime = (
                        self.X_prime[:, self.explain_indices]
                        * self.costs_vector_reshaped
                    )

                    self.term1 += (
                        mu[i, j]
                        * (
                            torch.dot(theta, weighted_X[i])
                            - torch.dot(theta, weighted_X_prime[j])
                        )
                        ** 2
                    )
        self.term1 /= torch.tensor(
            self.swd.n_proj, dtype=torch.float, device=self.device
        )

        # Compute the second term
        self.term2 = torch.tensor(0.0, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                self.term2 += (
                    nu[i, j] * (self.model(self.X[i]) - self.y_prime[j]) ** 2
                ).item()

        self.Q = (1 - eta) * self.term1 + eta * self.term2

    def _update_X_grads(self, mu_list, nu, eta, tau):
        n, m = (
            self.X[:, self.explain_indices].shape[0],
            self.X_prime[:, self.explain_indices].shape[0],
        )
        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]

        # Obtain model gradients with a dummy backward pass
        outputs = self.model(self.X)
        loss = outputs.sum()

        # Ensure gradients are zeroed out before backward pass
        self.X.grad = None
        loss.backward()
        model_grads = self.X.grad[
            :, self.explain_indices
        ].clone()  # Store the gradients

        # Weights applied to the features of X and X_prime
        weighted_X = self.X[:, self.explain_indices] * self.costs_vector_reshaped
        weighted_X_prime = (
            self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped
        )

        # Compute the projections with the weighted features
        X_proj = torch.stack(
            [torch.matmul(weighted_X, theta) for theta in thetas],
            dim=1,
        )  # Shape: [n, num_thetas]
        X_prime_proj = torch.stack(
            [torch.matmul(weighted_X_prime, theta) for theta in thetas],
            dim=1,
        )  # Shape: [m, num_thetas]

        # Use broadcasting to compute differences for all i, j
        differences = (
            X_proj[:, :, None] - X_prime_proj.T[None, :, :]
        )  # Shape: [n, num_thetas, m]

        # Multiply by mu and sum over j
        gradient_term1_matrix = torch.stack(
            [mu.to(self.device) * differences[:, k, :] for k, mu in enumerate(mu_list)],
            dim=1,
        )  # [n, num_thetas, m]
        gradient_term1 = torch.sum(
            gradient_term1_matrix, dim=2
        )  # Shape [n, num_thetas]

        # Weight by theta to get the gradient
        gradient_term1 = torch.matmul(
            gradient_term1, torch.stack(thetas)
        )  # Shape [n, d]

        # Compute the second term
        diff_model = self.model(self.X).unsqueeze(1) - self.y_prime.reshape(
            len(self.y_prime), 1
        )
        nu = nu.to(self.device)

        self.nu = nu
        self.diff_model = diff_model
        self.model_grads = model_grads

        gradient_term2 = (nu.unsqueeze(-1) * diff_model * model_grads.unsqueeze(1)).sum(
            dim=1
        )

        self.Qx_grads = (1 - eta) * gradient_term1 + eta * gradient_term2
        # self.Qx_grads = gradient_term2
        self.X.grad.zero_()
        self.X.grad[:, self.explain_indices] = self.Qx_grads * tau

    def __perform_SGD(self, past_Qs, eta, tau):
        # Reset the gradients
        self.optimizer.zero_grad()

        # Compute the gradients for self.X[:, self.explain_indices]
        self._update_X_grads(
            mu_list=self.swd.mu_list,
            nu=self.wd.nu,
            eta=eta,
            tau=tau,
        )

        # Perform an optimization step
        self.optimizer.step()

        # 关键修复：将未优化的列重置回原始值
        # 只有 explain_indices 的列应该被修改，其他列保持不变
        all_indices = set(range(self.X.shape[1]))
        unchanged_indices = list(all_indices - set(self.explain_indices))

        if len(unchanged_indices) > 0:
            with torch.no_grad():
                self.X[:, unchanged_indices] = self.X_prime[:, unchanged_indices]

        # Update the Q value, X_all, and y by the newly optimized X
        self._update_Q(mu_list=self.swd.mu_list, nu=self.wd.nu, eta=eta)
        self.y = self.model(self.X)

        # Check for convergence using moving average of past Q changes
        past_Qs.pop(0)
        past_Qs.append(self.Q.item())
        avg_Q_change = (past_Qs[-1] - past_Qs[0]) / 5
        return avg_Q_change

    def optimize_without_chance_constraints(
        self,
        eta=0.9,
        max_iter: Optional[int] = 100,
        tau=10,
        tol=1e-6,
        silent=True,
    ):
        logger.info("Optimization (without chance constraints) started")
        past_Qs = [float("inf")] * 5  # Store the last 5 Q values for moving average
        for i in tqdm(range(max_iter)):
            self.swd.distance(
                X_s=self.X[:, self.explain_indices] * self.costs_vector_reshaped,
                X_t=self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped,
                delta=self.delta,
            )
            self.wd.distance(y_s=self.y, y_t=self.y_prime, delta=self.delta)

            avg_Q_change = self.__perform_SGD(past_Qs, eta=eta, tau=tau)

            if not silent:
                logger.info(
                    f"Iter {i+1}: Q = {self.Q}, term1 = {self.term1}, term2 = {self.term2}"
                )

            if abs(avg_Q_change) < tol:
                logger.info(f"Converged at iteration {i+1}")
                break

        self.best_X = self.X.clone().detach()
        self.best_y = self.y.clone().detach()

    def optimize(
        self,
        U_1: float,
        U_2: float,
        alpha=0.05,
        l=0.2,
        r=1,
        kappa=0.05,
        max_iter: Optional[int] = 100,
        tau=10,
        tol=1e-6,
        bootstrap=True,
        silent=True,
    ):
        self.interval_left = l
        self.interval_right = r

        logger.info("Optimization started")
        past_Qs = [float("inf")] * 5  # Store the last 5 Q values for moving average
        for i in tqdm(range(max_iter)):
            swd_dist, _ = self.swd.distance(
                X_s=self.X[:, self.explain_indices] * self.costs_vector_reshaped,
                X_t=self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped,
                delta=self.delta,
            )
            wd_dist, _ = self.wd.distance(
                y_s=self.y,
                y_t=self.y_prime,
                delta=self.delta,
            )
            self.Qv_lower, self.Qv_upper = self.wd.distance_interval(
                self.y, self.y_prime, delta=self.delta, alpha=alpha, bootstrap=bootstrap
            )
            self.Qu_lower, self.Qu_upper = self.swd.distance_interval(
                self.X[:, self.explain_indices] * self.costs_vector_reshaped,
                self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped,
                delta=self.delta,
                alpha=alpha,
                bootstrap=False,
            )

            if not self.Qu_upper >= 0:
                self.Qu_upper = swd_dist

            if not self.Qv_upper >= 0:
                self.Qv_upper = wd_dist

            (
                eta,
                self.interval_left,
                self.interval_right,
            ) = self._get_eta_interval_narrowing(
                U_1=U_1,
                U_2=U_2,
                Qu_upper=self.Qu_upper,
                Qv_upper=self.Qv_upper,
                l=self.interval_left,
                r=self.interval_right,
                kappa=kappa,
            )

            self.wd_list.append(wd_dist)
            self.swd_list.append(swd_dist)
            self.wd_lower_list.append(self.Qv_lower)
            self.wd_upper_list.append(self.Qv_upper)
            self.swd_lower_list.append(self.Qu_lower)
            self.swd_upper_list.append(self.Qu_upper)
            self.eta_list.append(eta)
            self.interval_left_list.append(self.interval_left)
            self.interval_right_list.append(self.interval_right)

            if not silent:
                logger.info(
                    f"U_1-Qu_upper={U_1-self.Qu_upper}, U_2-Qv_upper={U_2-self.Qv_upper}"
                )
                logger.info(
                    f"eta={eta}, l={self.interval_left}, r={self.interval_right}"
                )

            avg_Q_change = self.__perform_SGD(past_Qs, eta=eta, tau=tau)

            if (U_1 - self.Qu_upper) < 0 or (U_2 - self.Qv_upper) < 0:
                gap = np.inf
            else:
                gap = (U_1 - self.Qu_upper) + (U_2 - self.Qv_upper)

            if gap < self.best_gap:
                self.best_gap = gap
                self.best_X = self.X.clone().detach()
                self.best_y = self.y.clone().detach()
                self.found_feasible_solution = True

            if not silent:
                logger.info(
                    f"Iter {i+1}: Q = {self.Q}, term1 = {self.term1}, term2 = {self.term2}"
                )

            if abs(avg_Q_change) < tol:
                logger.info(f"Converged at iteration {i+1}")
                break

        if not self.found_feasible_solution:
            self.best_gap = gap
            self.best_X = self.X.clone().detach()
            self.best_y = self.y.clone().detach()

    def _get_eta_set_shrinking(self):
        return 0.99

    def _get_eta_interval_narrowing(
        self, U_1, U_2, Qu_upper, Qv_upper, l=0, r=1, kappa=0.05
    ):
        """
        Implements the interval narrowing algorithm.

        Parameters:
        Qv_upper, Qu_upper (float): Upper confidence limits.
        l, r (float): Current lower and upper bounds of the interval.
        kappa (float): Contraction factor for the interval.

        Returns:
        eta (float): The point in the interval [l, r] that maximizes the objective function.
        l, r (float): Updated lower and upper bounds of the interval.
        """

        if not math.isfinite(Qv_upper):
            return l, l, r

        if not math.isfinite(Qu_upper):
            return r, l, r

        eta = self.__choose_eta_within_interval(
            a=U_1 - Qu_upper, b=U_2 - Qv_upper, l=l, r=r
        )

        # Narrow the interval
        if eta > (l + r) / 2:
            l = l + kappa * (r - l)
        else:
            r = r - kappa * (r - l)
        return eta, l, r

    def __choose_eta_within_interval(self, a, b, l, r):
        if (a < 0 and b >= 0) or (a >= 0 and b < 0):
            return l if a < 0 else r
        else:
            # For a, b both negative or both positive
            if a < 0 and b < 0:
                # Both negative: more weight to the more negative
                eta_proportion = b / (a + b)
            else:
                # Both positive: more weight to the less positive
                eta_proportion = a / (a + b)

            # Scale eta to be within the range [l, r]
            return l + eta_proportion * (r - l)

    def get_nu(self):
        return self.wd.nu.detach().numpy()

    def get_mu(self, method="avg"):

        if method == "avg":
            mu_avg = torch.zeros_like(self.swd.mu_list[0])
            for mu in self.swd.mu_list:
                mu_avg += mu

            total_sum = mu_avg.sum()

            matrix_mu = mu_avg / total_sum
            return matrix_mu
        else:
            raise NotImplementedError



class WassersteinDivergence:
    def __init__(self, reg=1):
        self.nu = None
        self.reg = reg

    def distance(self, y_s: torch.tensor, y_t: torch.tensor, delta):
        # Validate delta
        if delta < 0 or delta > 0.5:
            raise ValueError("Delta should be between 0 and 0.5")

        y_s = y_s.squeeze()
        y_t = y_t.squeeze()

        # Calculate quantiles
        lower_quantile_s = torch.quantile(y_s, delta)
        upper_quantile_s = torch.quantile(y_s, 1 - delta)
        lower_quantile_t = torch.quantile(y_t, delta)
        upper_quantile_t = torch.quantile(y_t, 1 - delta)

        # Indices in the original tensors that correspond to the filtered values
        indices_s = torch.where((y_s >= lower_quantile_s) & (y_s <= upper_quantile_s))[
            0
        ]
        indices_t = torch.where((y_t >= lower_quantile_t) & (y_t <= upper_quantile_t))[
            0
        ]

        # Create a meshgrid to identify the locations in self.nu to be updated
        indices_s_grid, indices_t_grid = torch.meshgrid(
            indices_s, indices_t, indexing="ij"
        )

        # Filter data points
        y_s_filtered = y_s[indices_s]
        y_t_filtered = y_t[indices_t]

        proj_y_s_dist_mass = torch.ones(len(y_s_filtered)) / len(y_s_filtered)
        proj_y_t_dist_mass = torch.ones(len(y_t_filtered)) / len(y_t_filtered)

        trimmed_M_y = ot.dist(
            y_s_filtered.reshape(y_s_filtered.shape[0], 1),
            y_t_filtered.reshape(y_t_filtered.shape[0], 1),
            metric="sqeuclidean",
        ).to("cpu")

        trimmed_nu = ot.emd(proj_y_s_dist_mass, proj_y_t_dist_mass, trimmed_M_y)
        # trimmed_nu = ot.bregman.sinkhorn(
        #     proj_y_s_dist_mass, proj_y_t_dist_mass, M_y, reg=self.reg
        # )
        # trimmed_nu = torch.diag(torch.ones(len(y_s)))
        dist = torch.sum(trimmed_nu * trimmed_M_y) * (1 / (1 - 2 * delta))

        self.nu = torch.zeros(len(y_s), len(y_t))

        # Place the values of trimmed_nu in the correct positions in self.nu
        self.nu[indices_s_grid, indices_t_grid] = trimmed_nu

        return dist, self.nu

    def distance_interval(
        self,
        y_s: torch.tensor,
        y_t: torch.tensor,
        delta: float,
        alpha: Optional[float] = 0.05,
        bootstrap=True,
    ):
        if bootstrap:
            return bootstrap_1d(
                y_s.detach().numpy(), y_t.detach().numpy(), delta=delta, alpha=alpha
            )
        else:
            return exact_1d(
                y_s.detach().numpy(), y_t.detach().numpy(), delta=delta, alpha=alpha
            )


class SlicedWassersteinDivergence:
    def __init__(self, dim: int, n_proj: int, reg=1):
        self.dim = dim
        self.n_proj = n_proj
        # self.thetas = np.random.randn(n_proj, dim)
        # self.thetas /= np.linalg.norm(self.thetas, axis=1)[:, None]

        # sample from the unit sphere
        self.thetas = np.random.multivariate_normal(
            np.repeat(0, dim), np.identity(dim), size=n_proj
        )
        self.thetas = np.apply_along_axis(
            lambda x: x / np.linalg.norm(x), 1, self.thetas
        )

        self.wd = WassersteinDivergence()

        self.reg = reg

        self.mu_list = []

    def distance(self, X_s: torch.tensor, X_t: torch.tensor, delta):
        """
        Compute the sliced Wasserstein distance between X_s and X_t

        Parameters:
        X_s : np.ndarray, shape (n_samples_a, dim)
            samples in the source domain
        X_t : np.ndarray, shape (n_samples_b, dim)
            samples in the target domain
        metric : str, optional
            metric to be used for Wasserstein-1 distance computation

        Returns:
        swd : float
            Sliced Wasserstein Distance between X_s and X_t
        """

        self.mu_list = []
        dist = 0
        for theta in self.thetas:
            # Project data onto the vector theta
            theta = torch.from_numpy(theta).float()
            proj_X_s = X_s.to("cpu") @ theta
            proj_X_t = X_t.to("cpu") @ theta

            dist_wd, mu = self.wd.distance(proj_X_s, proj_X_t, delta)

            self.mu_list.append(mu)

            dist += dist_wd

        return dist / self.n_proj, self.mu_list

    def distance_interval(
        self,
        X_s: torch.tensor,
        X_t: torch.tensor,
        delta: float,
        alpha: Optional[float] = 0.05,
        bootstrap=True,
    ):
        if bootstrap:
            return bootstrap_sw(X_s, X_t, delta=delta, alpha=alpha, swd=self)
        else:
            N = len(self.thetas)
            low = []
            up = []
            for theta in self.thetas:
                # Project data onto the vector theta
                theta = torch.from_numpy(theta).float()
                proj_X_s = X_s.to("cpu") @ theta
                proj_X_t = X_t.to("cpu") @ theta

                l, u = self.wd.distance_interval(
                    proj_X_s, proj_X_t, delta=delta, alpha=alpha / N
                )

                low.append(np.power(l, 2))
                up.append(np.power(u, 2))

            left = np.power(np.mean(low), 1 / 2)
            right = np.power(np.mean(up), 1 / 2)

            return left, right

def bootstrap_1d(x, y, delta, alpha, r=2, B=200):
    x = torch.tensor(x, dtype=torch.float32).squeeze()
    y = torch.tensor(y, dtype=torch.float32).squeeze()

    n, m = x.shape[0], y.shape[0]

    wd = WassersteinDivergence()
    dist_what, _ = wd.distance(x, y, delta)
    dist_what = dist_what.detach().numpy()

    # Generate all bootstrap indices at once
    x_indices = np.random.choice(n, (B, n))
    y_indices = np.random.choice(m, (B, m))

    W = np.empty(B)
    for i in range(B):
        xx = x[x_indices[i]]
        yy = y[y_indices[i]]

        dist, _ = wd.distance(xx, yy, delta)
        dist = dist.detach().numpy()
        W[i] = dist - dist_what

    q1, q2 = np.quantile(W, [alpha / 2, 1 - alpha / 2])

    Wlower = np.maximum(dist_what - q2, 0)
    Wupper = dist_what - q1

    if Wupper < 0:
        return 0, 0

    return np.power(Wlower, 1 / r), np.power(Wupper, 1 / r)
def exact_1d(x, y, delta, alpha, r=2, mode="DKW", nq=1000):
    """Confidence intervals for W_{r,delta}(P, Q) in one dimension.

    Parameters
    ----------
    x : np.ndarray (n,)
        sample from P
    y : np.ndarray (m,)
        sample from Q
    r : int, optional
        order of the Wasserstein distance
    delta : float, optional
        trimming constant, between 0 and 0.5.
    alpha : float, optional
        number between 0 and 1, such that 1-alpha is the level of the confidence interval
    mode : str, optional
        either "DKW" to use a confidence interval based on the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality [1,2]
        or "rel_VC" to use a confidence interval based on the relative Vapnik-Chervonenkis (VC) inequality [3]
    nq : int, optional
        number of quantiles to use in Monte Carlo integral approximations

    Returns
    -------
    l : float
        lower confidence limit

    u : float
        upper confidence limit

    References
    ----------

    .. [1] Dvoretzky, Aryeh, Jack Kiefer, and Jacob Wolfowitz.
           "Asymptotic minimax character of the sample distribution function and
           of the classical multinomial estimator." The Annals of Mathematical Statistics (1956): 642-669.

    .. [2] Massart, Pascal. "The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality." The annals of Probability (1990): 1269-1283.

    .. [3] Vapnik, V., Chervonenkis, A.: On the uniform convergence of relative frequencies of events to
           their probabilities. Theory of Probability and its Applications 16 (1971) 264–280.

    """
    x = x.squeeze()
    y = y.squeeze()
    us = np.linspace(delta, 1 - delta, nq)

    if mode == "DKW":
        try:
            Lx, Ux = aux._dkw(x, us, alpha)
            Ly, Uy = aux._dkw(y, us, alpha)

        except OverflowError:
            return (0, np.Inf)

    elif mode == "rel_VC":
        try:
            Lx, Ux = aux._rel_vc(x, us, alpha)
            Ly, Uy = aux._rel_vc(y, us, alpha)

        except OverflowError:
            return (0, np.Inf)

    elif mode == "sequential":
        Lx, Ux = aux._quantile_seq(x, us, delta=alpha)[-1, :]
        Ly, Uy = aux._quantile_seq(y, us, delta=alpha)[-1, :]

    else:
        raise Exception("Mode unrecognized.")

    low = np.repeat(0, nq)
    up = np.repeat(0, nq)

    low = np.fmax(Lx - Uy, Ly - Ux)
    low = np.power(np.fmax(low, np.repeat(0, nq)), r)
    up = np.power(np.fmax(Ux - Ly, Uy - Lx), r)

    lower_final = np.power((1 / (1 - 2 * delta)) * np.mean(low), 1 / r)
    upper_final = np.power((1 / (1 - 2 * delta)) * np.mean(up), 1 / r)

    return lower_final, upper_final
def bootstrap_sw(x, y, delta, alpha, swd, r=2, B=200):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    n, m, d = x.shape[0], y.shape[0], x.shape[1]

    SW_hat, _ = swd.distance(x, y, delta)
    SW_hat = SW_hat.detach().numpy()

    # Generate all bootstrap indices at once
    x_indices = np.random.choice(n, (B, n))
    y_indices = np.random.choice(m, (B, m))

    boot = np.empty(B)
    for i in range(B):
        xx = x[x_indices[i], :]
        yy = y[y_indices[i], :]

        SW_boot, _ = swd.distance(xx, yy, delta)
        SW_boot = SW_boot.detach().numpy()
        boot[i] = SW_boot - SW_hat

    q1, q2 = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    SW_lower = np.maximum(SW_hat - q2, 0)
    SW_upper = SW_hat - q1

    return np.power(SW_lower, 1 / r), np.power(SW_upper, 1 / r)
