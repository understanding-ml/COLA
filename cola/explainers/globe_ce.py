# The code originates from https://github.com/danwley/GLOBE-CE/blob/main/globe_ce.py
# Paper for reference:
# Ley, Dan, Saumitra Mishra, and Daniele Magazzeni.
# "GLOBE-CE: a translation based approach for global counterfactual explanations.
# " International Conference on Machine Learning. PMLR, 2023.
import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class GLOBE_CE:
    def __init__(
        self,
        model,
        dataset,
        X,
        affected_subgroup=None,
        dropped_features=[],
        ordinal_features=[],
        delta_init="zeros",
        normalise=None,
        bin_widths=None,
        monotonicity=None,
        p=1,
    ):
        """GLOBE_CE class. This class is used to generate GCE directions and evaluate scaling.

        (required arguments)
        model     : Contains predict function
        dataset   : Custom dataset wrapper that includes the data (there is no direct need to
                    pass x_aff as an argument) as well as categorical/continuous features information
        x_aff     : Pandas DataFrame. Inputs in training data which received a negative prediction

        (optional arguments)
        lr        : learning rate for gradient descent optimizer
        lams      : hyperparameters for objective function (currently using softmax prediction
                    + l1 distance regularization)
        delta_init: initial global translation before optimization is performed (default 0)
        cuda      : if GPU is to be used
        """
        # Params
        self.x_dim = X.shape[1]  # dimensionality of inputs
        self.p = p

        # Model + Dataset + Cuda
        self.model = model
        self.dataset = copy.deepcopy(dataset)
        self.X = copy.deepcopy(X)
        self.affected_subgroup = affected_subgroup
        self.name = self.dataset.name
        self.monotonicity = np.array(monotonicity) if monotonicity is not None else None
        self.features = np.array(list(self.dataset.features_tree))
        print(self.features)
        self.n_f = len(self.features)
        self.feature_values = self.dataset.features[:-1]

        # Refer normalisation to model?
        if normalise is not None:
            self.normalise = True
            self.means = normalise[0]
            self.stds = normalise[1]
            self.preds = self.model.predict((self.X.values - self.means) / self.stds)
        else:
            self.normalise = False
            self.preds = self.model.predict(X.values)  # to determine affected inputs

        # X
        self.x_aff = copy.deepcopy(self.X.values)
        self.x_aff = self.x_aff[self.preds == 0]
        if self.affected_subgroup is not None:
            self.subgroup_idx = self.x_aff[self.affected_subgroup] == 1
            self.x_aff = self.x_aff[self.subgroup_idx]
        self.n = self.x_aff.shape[0]  # number of affected inputs

        # Feature processing
        self.dropped_features = dropped_features
        self.active_idx = np.ones(len(self.feature_values), dtype=bool)
        self.active_f_idx = np.ones(self.n_f, dtype=bool)
        for i, feature_value in enumerate(self.feature_values):
            feature = feature_value.split()[0]
            f_i = self.features == feature
            if feature in self.dropped_features:
                self.active_idx[i] = False
                self.active_f_idx[f_i] = False
            if self.affected_subgroup is not None:
                if feature == self.affected_subgroup.split()[0]:
                    self.active_idx[i] = False
                    self.active_f_idx[f_i] = False
        self.sample_idx = np.arange(self.n_f)[self.active_f_idx]  # for random sampling

        # Store Features. Generate l1 feature costs (need to differentiate
        # between categorical/continuous)
        # Continuous/categorical/non-dropped features are all computed/stored
        # Parts of this section are copied from AReS (thus, all of these
        # variables may not be necessary)
        # Note that many variables are useful in debugging (can inspect
        # instance.variable from Jupyter)
        self.features_tree = (
            self.dataset.features_tree
        )  # dictionary of form 'feature: [feature values]'
        # list of categorical features (not values)
        self.categorical_features = self.dataset.categorical_features[self.name]
        # list of continuous features
        self.continuous_features = self.dataset.continuous_features[self.name]
        # Number of categorical or continuous features
        self.n_categorical = len(self.categorical_features)
        self.n_continuous = len(self.continuous_features)
        for feature in self.dropped_features:
            if feature in self.categorical_features:
                self.n_categorical -= 1
            else:
                self.n_continuous -= 1
        # Compute Costs Mask
        self.ordinal_features = ordinal_features
        self.feature_costs_vector = np.zeros(len(self.feature_values))
        self.non_ordinal_categories_idx = np.ones(len(self.feature_values), dtype=bool)
        self.bin_widths = bin_widths
        # Mask to clamp categorical variables (this is yet to be tested)
        self.categorical_idx = np.zeros(len(self.feature_values), dtype=bool)
        # Costs Masks and Feature Idx to Values Indexes Dictionary
        i = 0
        self.features_tree_idx = {}
        for j, feature in enumerate(self.features_tree):
            if feature in self.continuous_features:
                if self.bin_widths is not None:
                    # includes dropped features
                    self.feature_costs_vector[i] = 1 / self.bin_widths[feature]
                else:
                    self.feature_costs_vector[i] = 1
                self.non_ordinal_categories_idx[i] = True
                self.features_tree_idx[j] = [i]
                i += 1
            else:
                n = len(self.features_tree[feature])
                if feature in self.ordinal_features:
                    # default (for now) to unit change between bins
                    self.feature_costs_vector[i : i + n] = np.arange(n)
                    self.non_ordinal_categories_idx[i : i + n] = False
                else:
                    # categorical features have cost 1 (2 changes of 0.5)
                    self.feature_costs_vector[i : i + n] = 0.5
                    self.non_ordinal_categories_idx[i : i + n] = True
                self.categorical_idx[i : i + n] = True
                self.features_tree_idx[j] = list(range(i, i + n))
                i += n
        self.continuous_idx = ~self.categorical_idx
        # either we bin continuous features before model training (ordinal categories)
        # or we don't (non-ordinal categories)
        self.ordinal_categories_idx = ~self.non_ordinal_categories_idx
        self.feature_costs_vector_no_ordinal = copy.deepcopy(self.feature_costs_vector)
        self.feature_costs_vector_no_ordinal[self.ordinal_categories_idx] = (
            0.5  # for sampling
        )
        self.any_non_ordinal = self.non_ordinal_categories_idx.any()
        self.any_ordinal = self.ordinal_categories_idx.any()
        self.n_categorical = sum(self.categorical_idx)

        # Initialise delta
        if type(delta_init) == str:
            if delta_init == "zeros":
                self.delta = np.zeros(self.x_dim)
        else:
            self.delta = copy.deepcopy(delta_init)

        self.correct_matrix, self.cost_matrix = None, None
        self.deltas, self.best_delta, self.deltas_div = None, None, None
        self.correct_vector, self.cost_vector = None, None
        self.correct_max, self.cost_max = None, None
        self.scalars = None

    def round_categorical(self, cf):
        """
        This function is used after the optimization to compute the actual counterfactual
        Currently not implemented for optimization: argmax will likely break gradient descent

        Input: counterfactuals computed using x_aff + global translation
        Output: valid counterfactuals where one_hot encodings are integers (0 or 1), not floats
        """
        ret = np.zeros(cf.shape)
        i = 0
        for feature in self.features:  # requires list to maintain correct order
            if not self.features_tree[feature]:
                ret[:, i] = cf[:, i]
                i += 1
            else:
                n = len(self.features_tree[feature])
                ret[
                    np.arange(ret.shape[0]), i + np.argmax(cf[:, i : i + n], axis=1)
                ] = 1
                i += n
        return ret

    def compute_costs(self, counterfactuals, x_aff=None):
        """Compute the costs of the counterfactuals"""
        if x_aff is None:
            x_aff = self.x_aff.copy()
        x_diff = counterfactuals - x_aff
        ret = 0
        if self.any_non_ordinal:
            # e.g. sum(abs([-0.5, 0.5])) going from one bin to another has cost 1
            # sum(abs(diff)) also applies to continuous features
            ret += np.linalg.norm(
                x_diff[:, self.non_ordinal_categories_idx]
                * self.feature_costs_vector[self.non_ordinal_categories_idx],
                axis=1,
                ord=1,
            )
        if self.any_ordinal:
            # e.g. abs(sum([1, -3])) going from 3rd bin to 1st bin has cost 2
            ret += np.abs(
                (
                    x_diff[:, self.ordinal_categories_idx]
                    * self.feature_costs_vector[self.ordinal_categories_idx]
                ).sum(1)
            )
        return ret

    def evaluate(
        self,
        delta,
        idxs=None,
        vector=True,
        none_type=None,
        x_aff=None,
        non_zero_costs=False,
    ):
        """
        Evaluate the performance of delta. Returns prediction/cost vectors

        Input: delta (numpy), idxs (numpy, optional), none_type (str, optional),
               vector (bool)
        Output: predictions, costs (0 or inf or nan where predictions are 0)
                return types are numpy if vector else floats
        """
        if x_aff is None:
            x_aff = self.x_aff

        if idxs is not None:
            x_aff = x_aff[idxs]

        # Evaluate CEs
        cost = np.zeros(x_aff.shape[0])
        ces = (
            self.round_categorical(x_aff + delta)
            if self.n_categorical
            else x_aff + delta
        )
        if self.normalise:
            correct = self.model.predict((ces - self.means) / self.stds)
        else:
            correct = self.model.predict(ces)
        if non_zero_costs:
            if self.n_categorical:
                cost = self.compute_costs(counterfactuals=ces, x_aff=x_aff)
            else:
                cost = np.linalg.norm(
                    delta * self.feature_costs_vector, ord=self.p
                ).item()
        else:
            if correct.any():
                if self.n_categorical:
                    cost[correct == 1] = self.compute_costs(
                        counterfactuals=ces[correct == 1], x_aff=x_aff[correct == 1]
                    )
                else:
                    cost[correct == 1] = np.linalg.norm(
                        delta * self.feature_costs_vector, ord=self.p
                    ).item()

        # Process costs and return values
        if not vector:
            if correct.any():
                return correct.mean() * 100, np.mean(cost[cost != 0])
            else:
                return correct.mean() * 100, 0.0
        if none_type == "inf":
            cost[cost == 0] = np.inf
        elif none_type == "nan":
            cost[cost == 0] = np.nan
        return correct * 100, cost

    def rules(self, delta=None, x_aff=None, categorical=False):
        """
        Return feature-wise dictionary of GCEs according to delta

        Input: delta (numpy), idxs (numpy, optional), none_type (str, optional),
               vector (bool)
        Output: predictions, costs (0 or inf or nan where predictions are 0)
                return types are numpy if vector else floats
        """
        if delta is None:
            delta = self.best_delta
        if x_aff is None:
            x_aff = self.x_aff.copy()
        rules = {}
        i = 0
        for feature in self.features_tree:
            if self.features_tree[feature] == []:
                if delta[i] != 0 and not categorical:
                    rules[feature] = ("+" if delta[i] > 0 else "") + str(delta[i])
                i += 1
            else:
                feature_values = self.features_tree[feature]
                n = len(feature_values)
                delta_f = delta[i : i + n].copy()
                if delta_f.any():
                    idx = np.argmax(delta_f)
                    r_idx = delta_f < delta_f[idx] - 1
                    x_f = x_aff[:, i : i + n]
                    if r_idx.any() and x_f[:, r_idx].any():
                        use_not = sum(r_idx) > len(r_idx) / 2
                        use_bracket = (
                            sum(~r_idx) > 1 if use_not else False
                        )  # sum(r_idx) > 1
                        if_vals = [
                            u.split("= ")[-1]
                            for (u, v) in zip(feature_values, r_idx)
                            if use_not ^ v
                        ]
                        if_vals = " or ".join(if_vals)
                        if_vals = ["(", if_vals, ")"] if use_bracket else [if_vals]
                        if use_not:
                            if_vals.insert(0, "Not ")
                        if_str = "".join(if_vals)
                        then_str = feature_values[idx].split("= ")[-1]
                        rules[feature] = f"If {if_str}, Then {then_str}"
                i += n
        return rules

    @staticmethod
    def monotonic(x, y):
        """
        Drops all x_i, y_i pairs (x and y have same length)
        which result in a decrease in y as x increases
        This function is used to flatten the coverage vs cost profile
        Consider moving functions like this to a universal src file

        Input: x (typically the cost vector, numpy)
               y (typically the coverage vector, numpy)
        Output: predictions, costs (0 or inf or nan where predictions are 0)
                return types are numpy if vector else floats
        """
        y = y[np.argsort(x)]
        x = np.sort(x)
        max_item = y[0]
        retain_idx = np.ones(y.shape[0], dtype=bool)
        for i, item in enumerate(y):
            if item < max_item:
                retain_idx[i] = False
            else:
                max_item = item
        return x[retain_idx], y[retain_idx]

    def lower_bounds_k(self, delta):
        """Returns the lower bounds for the k values of the GCEs,
        given the delta vector and according to the categorical features"""
        ks = np.zeros(delta.shape[0])
        i = 0
        for feature in self.features_tree:
            if self.features_tree[feature] == []:
                i += 1
            else:
                feature_values = self.features_tree[feature]
                n = len(feature_values)
                delta_f = delta[i : i + n].copy()
                if delta_f.any():
                    i_max, delta_f_max = np.argmax(delta_f), np.max(delta_f)
                    if delta_f_max == 0:  # Avoid division by zero
                        delta_f += 1
                        delta_f_max += 1
                    delta_f[i_max] = 0
                    k_f = 1 / (
                        delta_f_max - delta_f
                    )  # Compute lower_bounds_k for this feature
                    k_f[i_max] = np.nan  # Resolve division by zero prevention code
                    # translation is 0 (almost certainly because that feature was dropped)
                    ks[i : i + n] = k_f
                else:
                    ks[i : i + n] = np.nan
                i += n
        return ks

    def scale(
        self,
        delta,
        scalars="auto",
        disable_tqdm=False,
        x_aff=None,
        n_scalars=1000,
        vector=False,
        plot=False,
        none_type=None,
        eps=None,
        non_zero_costs=False,
    ):
        """Scale the delta vector by a scalar and return the coverage, cost, and scalars"""
        # scale by maximum k
        # perform bisection
        if eps is None:
            eps = 0
        if x_aff is None:
            x_aff = self.x_aff

        self.scalars = scalars
        # Compute scalars
        if type(scalars) == str and scalars == "auto":
            max_scalar = max(self.bisection(delta), 1)
            self.scalars = np.linspace(0, max_scalar, n_scalars)

        # Evaluate scaled delta
        n_scalars = len(self.scalars)
        if vector:
            corrects = np.zeros((n_scalars, x_aff.shape[0]))
            costs = np.zeros((n_scalars, x_aff.shape[0]))
        else:
            corrects = np.zeros(n_scalars)
            costs = np.zeros(n_scalars)

        for i, scalar in enumerate(tqdm(self.scalars, disable=disable_tqdm)):
            if ~np.isnan(scalar):
                corrects[i], costs[i] = self.evaluate(
                    scalar * delta + eps,
                    vector=vector,
                    none_type=none_type,
                    x_aff=x_aff,
                    non_zero_costs=non_zero_costs,
                )

        return corrects, costs, self.scalars

    def bisection(self, delta, thresh=99.9, iters=200, b_lim=100):
        """Returns the maximum scalar for which the coverage is above thresh"""
        # Takes in delta, returns the multiplier b which results in coverage ~ thresh
        # Uses a simple bisection interval search
        b = 1  # initial upper interval
        max_acc = 0  # maximum coverage
        max_b = 1  # upper interval at maximum coverage
        ces = self.round_categorical(self.x_aff + delta * b)
        if self.normalise:
            ces = (ces - self.means) / self.stds
        pred = self.model.predict(ces).mean() * 100
        while pred < thresh and b < b_lim:
            if pred > max_acc:
                max_b = b
            b *= 2
            ces = self.round_categorical(self.x_aff + delta * b)
            if self.normalise:
                ces = (ces - self.means) / self.stds
            pred = self.model.predict(ces).mean() * 100
        if pred > max_acc:
            max_b = b
        a = b / 2  # lower interval
        i = 0
        while i < iters:
            c = (a + b) / 2  # midpoint
            ces = self.round_categorical(self.x_aff + delta * c)
            if self.normalise:
                ces = (ces - self.means) / self.stds
            if self.model.predict(ces).mean() * 100 > thresh:
                b = c
            else:
                a = c
            i += 1

        # idx = min_costs <= thresh
        # idxs = np.arange(min_costs.shape[0])[idx]
        # min_costs = min_costs[idx]
        return b

    def cluster_continuous(self, costs, n_bins, thresh=np.inf, return_bins=False):
        """Clusters the continuous features according to the costs, returns scalar_idxs"""
        min_costs, min_costs_idxs = self.min_scalar_costs(
            costs, return_idxs=True, remove_nan=True
        )
        min_costs = min_costs[min_costs <= thresh]
        min_costs_idxs = min_costs_idxs[min_costs <= thresh]
        bins = pd.cut(min_costs, bins=n_bins, precision=32)
        code, count = np.unique(bins.codes, return_counts=True)
        rights = bins.categories.values.right
        scalar_idxs = np.zeros(len(rights), dtype=int)
        for i, right in enumerate(rights):
            idx = min_costs <= right
            scalar_idxs[i] = min_costs_idxs[idx][min_costs[idx].argmax()]
        if return_bins:
            return scalar_idxs, bins
        return scalar_idxs

    def cluster_by_costs(self, costs, n_bins, thresh=np.inf):
        """Clusters the continuous features according to the costs."""
        # combine with self.group()
        max_costs = np.zeros(costs.shape[0])
        for i in range(costs.shape[0]):
            if (costs[i] <= thresh).any():
                max_costs[i] = costs[i][costs[i] <= thresh].max()

        bins = pd.cut(
            max_costs, bins=n_bins, precision=32
        )  # can replace with min_costs
        code, count = np.unique(bins.codes, return_counts=True)
        rights = bins.categories.values.right

        max_costs = costs.max(1)
        max_scalar_idxs = np.zeros(len(code))
        cost = costs.copy()

        for i, c in enumerate(code):
            idxs = np.arange(max_costs.shape[0])[max_costs <= rights[c]]
            if idxs.any():
                max_scalar_idxs[i] = idxs[-1]
                x_idxs = costs[idxs[-1]] == 0
                if x_idxs.any():
                    cost[: idxs[-1] + 1] = 0
                    max_costs = cost[:, x_idxs].max(1)
                    max_costs[: idxs[-1] + 1] = np.inf
            else:
                max_scalar_idxs[i] = np.inf
        return max_scalar_idxs

    def evaluate_clustering(
        self,
        delta,
        scalars,
        max_scalar_idxs,
        costs=None,
        x_aff=None,
        print_outputs=True,
        vector=False,
        eps=0,
        latex_table=False,
    ):
        """Evaluates the clustering by computing the coverage and cost for each cluster"""
        if x_aff is not None:
            x_aff = x_aff.copy()
        else:
            x_aff = self.x_aff.copy()
        if costs is None:
            costs = np.zeros((scalars.shape[0], x_aff.shape[0]))
            for i, scalar_idx in enumerate(max_scalar_idxs):
                if scalar_idx != np.inf:
                    scalar_idx = int(scalar_idx)
                    costs[scalar_idx] = self.evaluate(
                        delta * (scalars[scalar_idx] + eps), x_aff=x_aff, vector=True
                    )[1]
        max_scalar_costs = np.zeros(x_aff.shape[0])
        max_scalar_costs[:] = np.inf
        costs_c = np.zeros(max_scalar_idxs.shape[0])
        corrects_c = np.zeros(max_scalar_idxs.shape[0])
        cor, avg_cost = 0, 0
        last_rules = {}
        for i, scalar_idx in enumerate(max_scalar_idxs):
            # print(i)
            if scalar_idx != np.inf:
                scalar_idx = int(scalar_idx)
                scalar_cost = costs[scalar_idx].copy()
                scalar_cost[scalar_cost == 0] = np.inf
                x_idx = (
                    scalar_cost < max_scalar_costs
                )  # input indexes of new or better costs
                if x_idx.any():
                    max_scalar_costs[x_idx] = scalar_cost[x_idx]
                    new_cor = cor + x_idx.sum()
                    new_avg_cost = max_scalar_costs[x_idx].mean()
                    costs_c[i], corrects_c[i] = (
                        max_scalar_costs[max_scalar_costs != np.inf].mean(),
                        new_cor / x_aff.shape[0] * 100,
                    )
                    if print_outputs:
                        # if i!=0: print()
                        new_accs, new_costs = round(
                            (new_cor - cor) / x_aff.shape[0] * 100, 2
                        ), round(new_avg_cost, 2)
                        total_accs, total_costs = round(
                            new_cor / x_aff.shape[0] * 100, 2
                        ), round(costs_c[i], 2)
                        if (
                            not latex_table
                        ):  # latex_table assumes purely categorical delta
                            print(
                                "\033[1m\n New Inputs:\t+{}%\t".format(new_accs)
                                + "New Inputs Cost:\t{})".format(new_costs)
                            )
                        x = x_aff[x_idx]
                        rules = self.rules(delta * (scalars[scalar_idx] + eps), x_aff=x)
                        j = 0
                        prefix = "" if latex_table else " Rules:\033[0m\t"
                        multiple_rules = False
                        for rule in rules:
                            r = rules[rule]
                            if self.features_tree[rule] == []:
                                c = self.feature_costs_vector[~self.categorical_idx]
                                c = c[self.continuous_features.index(rule)].item()
                                print(
                                    prefix,
                                    rule
                                    + ": {} ({})".format(
                                        round(float(r), 6), round(float(r) * c, 2)
                                    ),
                                )
                            else:
                                if latex_table:
                                    if (rule not in last_rules) or (
                                        r not in last_rules[rule]
                                    ):
                                        prefix = "\n" if multiple_rules else ""
                                        multiple_rules = True
                                        print(prefix + rule + ": " + r)
                                else:
                                    print(prefix, rule + ":", r)
                            prefix = "\n" if latex_table else "\t"
                        if latex_table:
                            print(
                                "& {}\% & {} & {}\% & {}\\\\\\midrule".format(
                                    new_accs, new_costs, total_accs, total_costs
                                )
                            )
                        else:
                            print(
                                "\033[1m(Coverage:\t{}%\t".format(total_accs)
                                + "Average Cost:\t\t{})\033[0m".format(total_costs)
                            )
                        last_rules = rules
                    cor = new_cor
                    avg_cost = max_scalar_costs[max_scalar_costs != np.inf].mean()
        if print_outputs and not latex_table:
            print(
                "\n\033[1mCoverage:\t{}%".format(round(cor / x_aff.shape[0] * 100, 4))
            )
            print("Average Cost:\t{}".format(round(avg_cost, 4)))
        if vector:
            return costs_c, corrects_c
        else:
            return avg_cost, cor

    def min_scalar_costs(self, costs, return_idxs=True, remove_nan=False, inf=False):
        min_costs = np.zeros(costs.shape[1])  # |X_aff|
        min_costs_idxs = np.zeros(costs.shape[1])
        min_costs_idxs[:] = np.nan
        for i in range(costs.shape[1]):
            if costs[:, i].any():
                min_costs[i] = costs[costs[:, i] != 0, i][0]
                min_costs_idxs[i] = np.arange(costs.shape[0])[costs[:, i] != 0][0]
                # query if accuracies of scaled deltas are monotonic (not necessarily true)
                # assert not (correctsc[:, i]-np.sort(correctsc[:, i])).any()
        n = np.sum(min_costs == 0)
        print(
            "\033[1mUnable to find recourse for {}/{} inputs\033[0m".format(
                n, costs.shape[1]
            )
        )
        if remove_nan:
            i = ~np.isnan(min_costs_idxs)
            min_costs, min_costs_idxs = min_costs[i], min_costs_idxs[i]
        if inf:
            min_costs[min_costs == 0] = np.inf
        if return_idxs:
            return min_costs, min_costs_idxs
        else:
            return min_costs

    @staticmethod
    def accuracy_cost_bounds(min_costs):
        n = min_costs.shape[0]
        min_costs = min_costs[min_costs != 0]
        min_costs = np.sort(min_costs)
        costs = np.zeros(min_costs.shape[0] + 1)
        corrects = np.zeros(min_costs.shape[0] + 1)
        # First element of each vector is 0
        for i in range(min_costs.shape[0]):
            corrects[i + 1] = (i + 1) / n * 100
            costs[i + 1] = min_costs[: i + 1].mean()
        return costs, corrects

    def select_n_deltas(self, n_div, plot=False):
        cors, coss, deltas = self.correct_matrix, self.cost_matrix, self.deltas

        # Get n diverse deltas (maximum coverage)
        self.deltas_div = np.zeros((n_div, self.x_aff.shape[1]))
        c = np.zeros(cors.shape[1])
        for i in range(n_div):
            c_max, j_max = 0, 0
            for j in range(cors.shape[0]):
                c_j = np.maximum(c, cors[j])
                if c_j.mean() > c_max:
                    c_max, j_max = c_j.mean(), j
            c = np.maximum(c, cors[j_max])
            self.deltas_div[i] = deltas[j_max]

    def sample(
        self,
        n_sample,
        magnitude=1,
        sparsity_power=1,
        idxs=None,
        n_features=None,
        disable_tqdm=False,
        plot=False,
        seed=None,
        scheme="random",
        dropped_features=[],
    ):
        n, x_dim = self.x_aff.shape
        if idxs is not None:
            n = idxs.sum()
        self.correct_matrix = np.zeros((n_sample, n))
        self.cost_matrix = np.zeros((n_sample, n))
        self.deltas = np.zeros((n_sample, x_dim))
        inactive = np.array([self.feature_values.index(i) for i in dropped_features])
        self.correct_vector, self.cost_vector = np.zeros(n_sample), np.zeros(n_sample)
        self.correct_max, self.cost_max = np.zeros(n_sample), np.zeros(n_sample)
        self.best_delta = np.zeros(x_dim)
        cor_max, cos_max = 0, 0

        if seed is not None:
            np.random.seed(seed)

        for i in tqdm(range(n_sample), disable=disable_tqdm):
            self.deltas[i] = self.schemes(scheme, magnitude, sparsity_power, n_features)
            if inactive.any():
                self.deltas[i, inactive] = 0
            self.correct_matrix[i], self.cost_matrix[i] = self.evaluate(
                self.deltas[i], vector=True, idxs=idxs
            )

            if self.correct_matrix[i].any():
                cos_idx = self.cost_matrix[i] != 0
                cor, cos = (
                    self.correct_matrix[i].mean(),
                    self.cost_matrix[i, cos_idx].mean(),
                )
                if (cor > cor_max) or (cor == cor_max and cos < cos_max):
                    cor_max, cos_max, self.best_delta = cor, cos, self.deltas[i]
            else:
                cor, cos = 0, 0
            self.correct_vector[i], self.cost_vector[i] = cor, cos
            self.correct_max[i], self.cost_max[i] = cor_max, cos_max

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=3, dpi=200)
            fig.set_figwidth(11)
            fig.set_figheight(3)
            plt.subplots_adjust(wspace=0.3)
            ax[0].scatter(self.cost_vector, self.correct_vector)
            ax[0].set_title("Coverage vs Cost")
            ax[0].set_xlabel("Cost")
            ax[0].set_ylabel("Coverage (%)")
            ax[0].scatter(cos_max, cor_max, label=r"Best $\delta$")
            ax[0].legend(loc="upper left", fontsize=9)
            ax[1].plot(np.arange(1, n_sample + 1), self.correct_max)
            ax[1].set_title("Best Coverage vs\nNumber of Samples")
            ax[1].set_xlabel("Number of Samples")
            ax[1].set_ylabel("Best Coverage (%)")
            delta_cost = self.best_delta * self.feature_costs_vector
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            i, j = 0, 0
            for feature in self.features_tree:
                if not self.features_tree[feature]:
                    ax[2].bar(
                        range(i, i + 1),
                        delta_cost[i],
                        hatch="/",
                        linewidth=1,
                        edgecolor="black",
                        color=cycle[j % len(cycle)],
                    )
                    i += 1
                    j += 1
                else:
                    feature_values = self.features_tree[feature]
                    n = len(feature_values)
                    ax[2].bar(
                        range(i, i + n),
                        delta_cost[i : i + n],
                        color=cycle[j % len(cycle)],
                    )
                    i += n
                    j += 1
            ax[2].set_title("Feature Cost\n" + r"Profile of Best $\delta$")
            ax[2].set_xlabel("Feature Index")
            ax[2].set_ylabel("Cost")
            plt.show()

    def schemes(self, scheme, magnitude=1, sparsity_power=1, n_features=None):
        # should be more general:
        # a) how many features e.g. n_features = 10
        # b) which features e.g. random, highest cost, specific features
        #    e.g. highest feature importance
        # c) magnitudes for features e.g. random, feature importance weighted
        # d) sparsity power, magnitude
        if n_features is None:
            n_features = len(self.sample_idx)

        if scheme == "random":  # n_features, random, random, (m, s)
            delta = np.zeros(self.x_dim)
            idx = np.random.choice(self.sample_idx, n_features, replace=False)
            s_idx, n_f = [], 0
            for i in idx:
                s_idx += self.features_tree_idx[i]
                n_f += len(self.features_tree_idx[i])
            s_idx = np.array(s_idx)
            delta[s_idx] = np.random.rand(n_f) ** sparsity_power
            if self.monotonicity is not None:
                delta *= self.monotonicity
            if self.n_continuous:
                delta[self.continuous_idx] *= (
                    np.random.choice(2, self.n_continuous) * 2 - 1
                )
            delta = (
                delta
                / np.linalg.norm(delta, ord=1)
                * magnitude
                / self.feature_costs_vector_no_ordinal
            )

        elif scheme == "features":  # n, highest feature, random,  (m, s)
            delta = np.zeros(self.x_dim)
            features_scoring = self.model.get_booster().get_score(
                importance_type="gain"
            )
            scores = np.array(list(features_scoring.values()))
            idxs = np.random.choice(
                len(scores), size=n_features, p=scores / sum(scores)
            )
            f_idxs = np.array([int(i.split("f")[-1]) for i in features_scoring.keys()])
            delta[f_idxs[idxs]] = np.random.rand(n_features) ** sparsity_power
            if self.monotonicity is not None:
                delta *= self.monotonicity
            else:
                delta *= np.random.choice(2, self.x_dim) * 2 - 1
            delta = (
                delta
                / np.linalg.norm(delta, ord=1)
                * magnitude
                / self.feature_costs_vector_no_ordinal
            )

        return delta

    def evaluate_deltas(self, deltas, scalars, x_aff=None):
        if x_aff is None:
            x_aff = self.x_aff
        n_div = deltas.shape[0]
        min_costs_te = np.zeros((n_div, x_aff.shape[0]))
        for i in range(n_div):
            cor_s, cos_s = self.scale(
                deltas[i], scalars=scalars, disable_tqdm=False, vector=True, x_aff=x_aff
            )
            min_costs_te[i] = self.min_scalar_costs(cos_s, return_idxs=False, inf=True)
        min_costs_te = min_costs_te.min(axis=0)
        cos_bound_te, cor_bound_te = self.clustering_lower_bounds(min_costs_te)
        return cos_bound_te, cor_bound_te

    def rules_to_delta(self, rules):
        # not suitable for scaling
        delta = np.zeros(self.x_dim)
        exclude = {"If", "Then", "or"}
        for r in rules:
            if self.features_tree[r] == 0:
                delta[self.feature_values.index(r)] = float(rules[r])
            else:
                feature_values = self.features_tree[r]
                r_split = rules[r].split()
                ifs = [
                    i.strip(",").strip("(").strip(")")
                    for i in r_split
                    if i not in exclude
                ]
                then, ifs = ifs[-1], ifs[:-1]
                if ifs[0] == "Not":
                    idxs = [
                        self.feature_values.index(i)
                        for i in feature_values
                        if i.split()[-1] in ifs
                    ]
                else:
                    idxs = [
                        self.feature_values.index(i)
                        for i in feature_values
                        if i.split()[-1] not in ifs
                    ]
                delta[idxs] = 1
                delta[self.feature_values.index(r + " = " + then)] = 1.1
        return delta
