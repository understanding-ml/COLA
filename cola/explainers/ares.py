# The code originates from https://github.com/danwley/GLOBE-CE/blob/main/ares.py
# Paper for reference:
# Ley, Dan, Saumitra Mishra, and Daniele Magazzeni.
# "GLOBE-CE: a translation based approach for global counterfactual explanations.
# " International Conference on Machine Learning. PMLR, 2023.

from mlxtend.frequent_patterns import apriori
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pandas as pd
import copy
import matplotlib.pyplot as plt
import itertools
import warnings


def create_feature_values_tree(features_tree, use_values=False):
    """
    Converts feature values back to their parent features REFER TO EFFICIENCY
    (used in self.compute_V to determine if rules are valid)

    Input: features_tree (a dictionary where features are keys and feature values are values)
    Output: dictionary where keys are feature values and values are the parent feature e.g.
            {... 'Foreign-Worker = A201': Foreign-Worker, 'Foreign-Worker = A202': Foreign-Worker ...}.
    """
    feature_values_tree = {}
    if use_values:
        for feature_value in features_tree:
            feature_values_tree[feature_value] = feature_value.split(" = ")[0]
    else:
        for feature, feature_values in features_tree.items():
            for feature_value in feature_values:
                feature_values_tree[feature_value] = feature
    return feature_values_tree


def create_features_tree(feature_values):
    features_tree = {}
    for feature_value in feature_values:
        feature = feature_value.split(" = ")[0]
        if feature not in features_tree:
            features_tree[feature] = [feature_value]
        else:
            features_tree[feature] += [feature_value]
    return features_tree


class AReS:
    def __init__(
        self,
        model,
        dataset,
        X,
        dropped_features=[],
        n_bins=10,
        ordinal_features=[],
        normalise=False,
        constraints=[20, 7, 10],
        correctness=False,
    ):
        """
        Normalise is implemented differently to GLOBE_CE
        AReS Implementation:

        (required arguments)
        model            : Any black box model with a predict() method that
                           returns a binary prediction
        x_aff            : Pandas DataFrame. Full training data of interest
                           (positive and negative predictions)
        dataset          : Our custom dataset_loader object including the data
                           (there is no direct need to pass X as an argument)
                           and information on categorical/continuous features

        (optional arguments)
        dropped_features    : List of dropped features in the form of just the
                              feature e.g. 'Foreign-Worker'
        add_redundant       : If True, evaluate each candidate rule and reject
                              those which don't provide any recourse for the
                              affected inputs (speeds up optimisation)
        apriori_threshold   : The support threshold used by the apriori
                              algorithm (probability of an itemset, lower
                              values thus return more possible rules)
        constraints         : As defined by the paper
                              e1 = total number of rules
                              e2 = maximum rule width (number of conditions)
                              e3 = total number of unique subgroup descriptors
                                   (outer if-then clauses)
        lams                : hyperparameters for objective function (list of
                              size 4 for AReS, size 2 for our objective)
        feature_costs       : optional vector for defined feature costs
                              (otherwise, we use l1 norm)
        ordinal_features    : List of categorical features that require ordinal
                              costs when moving between categories (typically
                              continuous features which have been one-hot
                              encoded before model training)
        original_objective  : If True, use the original AReS objective function
                              (otherwise just optimise correctness and cost)
        n_bins              : number of (equal) bins for continuous variables
        normalise           : If True, normalise the inputs prior to the
                              self.model.predict() call
        then_generation     : Apriori threshold value. In progress. If not
                              None, then generate the "then" condition using
                              apriori on a set filtered according to each if-if
                              condition (search for candidate rules in SDxRL).
                              May find more relevant rules. If None, search for
                              candidate rules in SDxRLxRL. There's also the
                              possibility to set "then" to RL, and divide SD
                              appropriately to match RL. As well as
                              (alternatively) the possibility to allow "then"
                              to not match the "inner-if" entirely.
        """

        # Set Input Parameters
        self.model = model
        self.normalise = normalise
        if self.normalise:
            self.means = X.values.mean(axis=0)
            self.stds = X.values.std(axis=0)
            self.preds = self.model.predict((X.values - self.means) / self.stds)
        else:
            self.preds = self.model.predict(X.values)  # to determine affected inputs
        self.X_original = X  # store original inputs
        # copy is needed since continuous features are binned for apriori
        self.X = self.X_original.copy()
        self.dataset = copy.deepcopy(dataset)
        self.dropped_features = dropped_features
        self.ordinal_features = ordinal_features
        self.n_bins = n_bins
        self.correctness = correctness
        self.e1, self.e2, self.e3 = constraints

        # Store Features. Generate l1 feature costs (need to differentiate between categorical/continuous)
        # Continuous/categorical/non-dropped features are all computed/stored
        self.features_tree = (
            self.dataset.features_tree
        )  # dictionary of form 'feature: [feature values]'
        self.features_tree_dropped = copy.deepcopy(self.features_tree)
        for feature in self.dropped_features:
            del self.features_tree_dropped[feature]
        self.features = (
            self.dataset.features
        )  # list of feature values (includes class label)
        # list of categorical features (not values)
        # if a continuous feature was binned before model training, then it's treated as categorical (though ordinal)
        self.categorical_features = self.dataset.categorical_features[self.dataset.name]

        # Bin continuous features and store resulting data (dimensionality of input data increases)
        self.X, self.binned_features, self.binned_features_continuous = (
            self.bin_continuous_features(self.X)
        )
        self.continuous_features = []  # list of continuous features
        self.feature_costs_vector = np.zeros(len(self.features) - 1)
        self.non_ordinal_categories_idx = np.ones(len(self.features) - 1, dtype=bool)
        i = 0
        for feature in self.features_tree:
            if feature not in self.categorical_features:
                self.continuous_features.append(feature)
                self.feature_costs_vector[i] = (
                    1 / self.bin_widths[feature]
                )  # includes dropped features
                self.non_ordinal_categories_idx[i] = True
                i += 1
            else:
                n = len(self.features_tree[feature])
                if feature in self.ordinal_features:
                    self.feature_costs_vector[i : i + n] = range(
                        n
                    )  # if ordinal, default to unit change between bins
                    self.non_ordinal_categories_idx[i : i + n] = False
                else:
                    self.feature_costs_vector[i : i + n] = (
                        0.5  # categorical features have cost 1 (2 changes of 0.5)
                    )
                    self.non_ordinal_categories_idx[i : i + n] = True
                i += n
        # either we bin continuous features before model training (ordinal categories),
        # or we don't (non-ordinal categories)
        # non-continuous features are also included in non-ordinal categories
        # (see self.objective_terms r_costs computation)
        self.ordinal_categories_idx = ~self.non_ordinal_categories_idx
        self.any_non_ordinal = self.non_ordinal_categories_idx.any()
        self.any_ordinal = self.ordinal_categories_idx.any()

        # Drop features
        self.X_drop = (
            self.X.copy()
        )  # self.X_drop is used just for apriori itemset generation
        for feature in self.dropped_features:
            print("Dropping Feature:", feature)
            for feature_value in self.features_tree[feature]:
                self.X_drop = self.X_drop.drop(feature_value, axis=1)

        # Compute affected features
        self.X_aff_original = (
            self.X_original.iloc[self.preds == 0].copy().reset_index(drop=True)
        )  # original data
        self.X_aff = (
            self.X.iloc[self.preds == 0].copy().reset_index(drop=True)
        )  # data with continuous variables binned
        self.U = self.n_bins * self.e2  # custom objective function
        self.U1 = self.X_aff.shape[0] * self.e1  # incorrectrecourse
        self.U3 = 0  # featurecost, not implemented
        self.U4 = self.n_bins * self.e1 * self.e2  # featurechange

        # Assign features to feature values, used when computing if rules are valid
        self.feature_values_tree = create_feature_values_tree(
            self.features_tree, use_values=False
        )

        # The following are updated using
        # self.compute_SD_RL and self.compute_V
        self.SD, self.RL, self.RL2 = None, None, None
        self.SD_copy, self.RL_copy = None, None
        self.V, self.V_copy = None, None
        self.f, self.V_opt = None, None
        self.R = None

    def bin_continuous_features(self, data):
        """
        Method for binning continuous features. Also computes self.bin_mids and self.bin_mids_tree (dictionary
        and dictionary of dictionaries respectively) which store mid point values for each bin range

        Input: original data
        Outputs: data with continuous features binned (default is 10 equally sized bins)
                 list of all feature values, with binned feature values included
                 list of only binned feature values
        """
        self.data_binned = data.copy()
        data_oh, features, continuous_features = [], [], set()
        self.bins = {}
        self.bin_mids = {}
        self.bin_mids_tree = {}
        self.bin_widths = {}
        for x in data.columns:
            if x.split()[0] in self.categorical_features:
                data_oh.append(pd.DataFrame(data[x]))
                features.append(x)
            else:
                self.data_binned[x], self.bins[x] = pd.cut(
                    self.data_binned[x].apply(lambda x: float(x)),
                    bins=self.n_bins,
                    retbins=True,
                )
                one_hot = pd.get_dummies(self.data_binned[x])
                one_hot.columns = pd.Index(list(one_hot.columns))  # necessary?
                data_oh.append(one_hot)
                cols = self.data_binned[x].cat.categories
                self.bin_mids_tree[x] = {}
                width = cols.length[-1]
                self.bin_widths[x] = width
                for i, col in enumerate(cols):
                    feature_value = x + " = " + str(col)
                    features.append(feature_value)
                    continuous_features.add(feature_value)
                    self.features_tree[x].append(feature_value)
                    mid = (
                        cols.mid[1] - width if i == 0 else col.mid
                    )  # adjust for pd.cut extending the first bin
                    self.bin_mids[feature_value] = mid
                    self.bin_mids_tree[x][feature_value] = mid
        data_oh = pd.concat(data_oh, axis=1)
        data_oh.columns = features
        return data_oh, features, continuous_features

    def generate_itemsets(
        self, apriori_threshold, max_width=None, affected_subgroup=None, save_copy=False
    ):
        """
        affected_subgroup   : The feature value of the subgroup of interest
                              e.g. 'Foreign-Worker = A201' (see dataset_loader naming)
                              If None, SD and RL are set to the same set
                              generated by apriori
        """
        # Max width
        if max_width is None:
            max_width = self.e2 - 1
        # Compute SD and RL
        print("Computing Candidate Sets of Conjunctions of Predicates SD and RL")
        self.SD = Apriori(
            x=self.X_drop,
            apriori_threshold=apriori_threshold,
            affected_subgroup=affected_subgroup,
            max_width=max_width,
            feature_values_tree=self.feature_values_tree,
        )
        if affected_subgroup is None:
            self.RL = copy.deepcopy(self.SD)
        else:
            self.RL = Apriori(
                x=self.X_drop,
                apriori_threshold=apriori_threshold,
                affected_subgroup=None,
                max_width=max_width,
                feature_values_tree=self.feature_values_tree,
            )

        # Update affected inputs
        self.X_aff_original = (
            self.X_original.iloc[(self.preds == 0) & self.SD.sub_idx]
            .copy()
            .reset_index(drop=True)
        )  # original data
        self.X_aff = (
            self.X.iloc[(self.preds == 0) & self.SD.sub_idx]
            .copy()
            .reset_index(drop=True)
        )  # data with continuous variables binned

        print(
            "SD and RL Computed with Lengths {} and {}".format(
                self.SD.length, self.RL.length
            )
        )

        if save_copy:
            print("Saving Copies of SD and RL as SD_copy and RL_copy")
            self.SD_copy, self.RL_copy = copy.deepcopy(self.SD), copy.deepcopy(self.RL)

    def generate_groundset(
        self, max_width=None, RL_reduction=False, then_generation=None, save_copy=False
    ):
        """
        Compute candidate set of rules for self.optimise(). Determines if rules are valid and also applies
        maxwidth constraint. User sets self.add_redundant to False (__init__ method) if we ignore any rules
        that do not provide any successful recourse (slower, but completely irrelevant rules are not added).
        Size of candidate rules, V, seems to be the bottleneck in the submodular maximisation.

        Inputs: SD and RL: outer and inner if conditions (as per paper)
                SD_lengths and RL_lengths: widths of each SD/RL element
                feature_values_tree: as described in self.encode_feature_values
                then_gen UPDATE
        Output: candidate set of rules after applying constraints
        """
        # Max width
        if max_width is None:
            max_width = self.e2

        self.V = TwoLevelRecourseSet()

        self.V.generate_triples(
            self.SD,
            self.RL,
            max_width=max_width,
            RL_reduction=RL_reduction,
            then_generation=then_generation,
        )

        print("Ground Set Computed with Length", self.V.length)
        if save_copy:
            print("Saving Copy of Ground Set as V_copy")
            self.V_copy = copy.deepcopy(self.V)

    def evaluate_groundset(
        self, lams, r=None, save_mode=0, disable_tqdm=False, plot_accuracy=False
    ):
        self.V.evaluate_triples(
            self,
            r=r,
            save_mode=save_mode,
            disable_tqdm=disable_tqdm,
            plot_accuracy=plot_accuracy,
        )

        # compute objectives for individual triples
        if len(lams) == 2:
            self.V.objectives = AReS.f_custom(self.V, lams, self.U, singleton=True)
        else:
            bounds = [self.U1, self.U3, self.U4]
            self.V.objectives = AReS.f_ares(self.V, lams, bounds, singleton=True)

        # add correctness later if you cba

    def select_groundset(self, s=0):
        self.V.select_triples(s)
        # implement updated cumulative plot
        # if plot_accuracy:
        #     self.V.plot_accuracy()

    def optimise_groundset(
        self, lams, factor=1, print_updates=False, print_terms=False, save_copy=False
    ):
        """
        Submodular maximisation. We make 2 major modifications:
            1. Don't repeat procedure k times, where k is the number of constraints. This rarely increased
               performance yet increases computation time k-fold (mostly pointless despite formal guarantees)
            2. Don't permit up to k elements to be exchanged (computationally infeasible- to this day I am clueless
               regarding how this is done efficiently). In this case, you might have 20 choose 2 = 190 options for
               elements to exchange (instead of just 20) which is just not a worthwhile trade-off.

        Output: Final two level recourse set, S
        """
        print("Initialising Copy of Ground Set")
        if save_copy:
            self.V_opt = None
            # ensures python doesn't use an already
            # stored version during the deepcopy process
            self.V_opt = copy.deepcopy(self.V)
            print("Ground Set Copied")
        else:
            self.V_opt = self.V
        N = self.V_opt.length
        selected_idx = np.argsort(self.V_opt.objectives)[: -(N + 1) : -1]
        self.V_opt.objectives = self.V_opt.objectives[selected_idx]
        self.V_opt.triples_array = self.V_opt.triples_array[selected_idx]
        self.V_opt.index_terms(selected_idx)
        self.V_opt.cost_matrix[np.isnan(self.V_opt.cost_matrix)] = 0
        self.f = AReS.f_custom if len(lams) == 2 else AReS.f_ares

        R_idx = np.zeros(N, dtype=bool)
        f_argmax = np.argmax(self.V_opt.objectives)
        f_max = self.V_opt.objectives[f_argmax]
        R_idx[f_argmax] = True
        f_thresh = factor * f_max

        # Compute objectives then select triples
        if len(lams) == 2:
            bounds = self.U
            self.f = AReS.f_custom
        else:
            bounds = [self.U1, self.U3, self.U4]
            self.f = AReS.f_ares

        # While there exists a delete/update operation do:
        print("While there exists a delete/update operation, loop:")
        delete, add, exchange = True, True, True
        while True:
            # Delete check
            print("Checking Delete")
            delete = False
            for idx in np.arange(N)[R_idx]:
                R_idx[idx] = False
                f_delete = self.f(self.V_opt, lams, bounds, idx=R_idx)
                if f_delete > f_thresh:
                    if print_updates:
                        print("Deleting Element ({} >= {})".format(f_delete, f_thresh))
                    # self.min_costs.append(self.f_custom(Si_delete, return_costs=True))
                    if print_terms:
                        self.f(self.V_opt, lams, bounds, idx=R_idx, print_terms=True)
                    f_thresh = factor * f_delete
                    delete = True
                    break
                R_idx[idx] = True
            if (not delete) and print_updates:
                print("No Delete Operation Found")
            if not (delete or add or exchange):
                break

            # Actual flow should be: always add elements until
            # constraint is reached or no element to add
            # Then exchange up to k... ?

            # Add check
            print("Checking Add")
            add = False
            if R_idx.sum() < self.e1:
                for idx in tqdm(np.arange(N)[~R_idx]):
                    R_idx[idx] = True
                    if self.constraints(self.V_opt.triples_array[R_idx]):
                        f_add = self.f(self.V_opt, lams, bounds, idx=R_idx)
                        if f_add > f_thresh:
                            if print_updates:
                                print(
                                    "Adding Element ({} >= {})".format(f_add, f_thresh)
                                )
                            if print_terms:
                                self.f(
                                    self.V_opt,
                                    lams,
                                    bounds,
                                    idx=R_idx,
                                    print_terms=True,
                                )
                            # self.min_costs.append(self.f_custom(Si_add, return_costs=True))
                            f_thresh = factor * f_add
                            add = True
                            continue
                    R_idx[idx] = False
            if (not add) and print_updates:
                print("No Add Operation Found")
            if not (delete or add or exchange):
                break

            # Exchange check
            print("Checking Exchange")
            exchange = False
            for add_idx in tqdm(np.arange(N)[~R_idx]):
                # Permit only 1 removal (not k, as in algorithm)
                for delete_idx in np.arange(N)[R_idx]:
                    R_idx[add_idx] = True
                    R_idx[delete_idx] = False
                    if self.constraints(self.V_opt.triples_array[R_idx]):
                        f_exchange = self.f(self.V_opt, lams, bounds, idx=R_idx)
                        if f_exchange > f_thresh:
                            if print_updates:
                                print(
                                    "Exchanging Element ({} >= {})".format(
                                        f_exchange, f_thresh
                                    )
                                )
                            if print_terms:
                                self.f(
                                    self.V_opt,
                                    lams,
                                    bounds,
                                    idx=R_idx,
                                    print_terms=True,
                                )
                            # self.min_costs.append(self.f(Si_exchange, print_terms=True, return_costs=True))
                            # self.min_costs.append(self.f_custom(Si_exchange, return_costs=True))
                            f_thresh = factor * f_exchange
                            exchange = True
                            break
                    R_idx[delete_idx] = True
                    R_idx[add_idx] = False
            if (not exchange) and print_updates:
                print("No Exchange Operation Found")
            if not (delete or add or exchange):
                break
            # break  # fix this to loop multiple times but only if necessary. also skip "add" if size constraints met
        # self.min_costs = np.array(self.min_costs)
        # self.V -= Si

        self.R = TwoLevelRecourseSet()
        self.R.triples = set(self.V_opt.triples_array[R_idx])
        self.R.length = len(self.R.triples)
        self.R.evaluate_triples(self)

    @staticmethod
    def f_ares(
        tlrs, lams, bounds, idx=None, singleton=False
    ):  # , print_terms=False, plot_f=False):
        # tlrs = two level recourse set
        if idx is None:
            idx = np.ones(tlrs.correct_matrix.shape[0], dtype=bool)
        if not idx.any():
            return lams[0] * bounds[0]
        featurecost = tlrs.featurecost[idx]
        featurechange = tlrs.featurechange[idx]
        if singleton:
            incorrectrecourse = (tlrs.correct_matrix[idx] == 0).sum(axis=1)
            cover = tlrs.cover_matrix[idx].sum(axis=1)
        else:
            incorrectrecourse = (tlrs.correct_matrix[idx] == 0).sum()
            cover = tlrs.cover_matrix[idx].max(axis=0).sum()
            featurecost = featurecost.sum()
            featurechange = featurechange.sum()
        return (
            lams[0] * (bounds[0] - incorrectrecourse)
            + lams[1] * cover
            + lams[2] * (bounds[1] - featurecost)
            + lams[3] * (bounds[2] - featurechange)
        )
        # In the cost terms, AReS doesn't consider how many points
        # are affected by a certain rule (seems unreliable)
        # E.g. you could find one high cost rule that applies to
        # all points, then the rest all low costs? Our implementation
        # considers the actual magnitudes of cost per triple and input
        # if plot_f:
        #     f.append(self.lams[0] * (self.U1 - incorrectrecourse) + self.lams[1] * cover\
        #         + self.lams[2] * (self.U3 - featurecost) + self.lams[3] * (self.U4 - featurechange))
        #     plt.plot(range(len(f)), f)
        #     plt.show()
        # if print_terms:
        #     print("{}/{}".format((self.U1 - incorrectrecourse), self.U1),
        #           cover, "{}/{}".format((self.U4 - featurechange), self.U4))
        #     print("Acc: {}, Cost: {}".format(round(cor.mean()*100, 4),
        #                                      round(cos[cor == 1].mean(), 4)))

    @staticmethod
    def f_custom(
        tlrs, lams, bound, idx=None, singleton=False, print_terms=False
    ):  # , plot_f=False):
        # tlrs = two level recourse set
        if singleton and print_terms:
            raise ValueError(
                "Cannot use parameters singleton and " "print_terms simultaneously"
            )
        if idx is None:
            idx = np.ones(tlrs.correct_matrix.shape[0], dtype=bool)
        elif not idx.any():
            return 0
        if singleton:
            correct = tlrs.correct_matrix[idx].mean(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                cost = np.nanmean(tlrs.cost_matrix[idx], axis=1)
            cost[np.isnan(cost)] = 0
        else:
            correct = tlrs.correct_matrix[idx].max(axis=0).mean()
            if correct == 0:
                cost = bound
            else:
                # Suppress numpy warning for all nan slice
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    cost = np.nanmean(np.nanmin(tlrs.cost_matrix[idx], axis=0))
                    if np.isnan(cost):
                        cost = bound
        if print_terms:
            n = tlrs.correct_matrix.shape[1]
            print(
                "Accuracy: {}/{} = {}%".format(
                    int(correct * n / 100), n, round(correct, 2)
                ),
                "\nAverage Cost: {}".format(round(cost, 4)),
            )
            # "{}/{}".format(round(np.average(objectives[objectives!=0]),2), self.lams[1]*self.U))
        return lams[0] * correct + lams[1] * (bound - cost)

    def constraints(self, triples):
        """
        Computes if constraints (e1: total number of rules, e3: total number of unique sub-descriptors) are violated

        Input: Two Level Recourse Set, Si
        Output: boolean (True if constraints are not violated)
        """
        if len(triples) > self.e1:
            return False
        subgroups = set()
        for triple in triples:
            subgroups.add(triple[0])  # only adds if the sub-descriptor is unseen
        if len(subgroups) > self.e3:
            return False
        return True

    def bin_X_test(self, data):
        """
        Combine with first class method?
        """
        label_encoder = preprocessing.LabelEncoder()
        data_encode = data.copy()
        self.data_binned_te = data.copy()
        data_oh = []
        for x in data.columns:
            if x.split()[0] in self.categorical_features:
                data_oh.append(pd.DataFrame(data[x]))
            else:
                self.data_binned_te[x] = pd.cut(
                    self.data_binned_te[x].apply(lambda x: float(x)), bins=self.bins[x]
                )
                one_hot = pd.get_dummies(self.data_binned_te[x])
                one_hot.columns = pd.Index(list(one_hot.columns))  # necessary?
                data_oh.append(one_hot)
        data_oh = pd.concat(data_oh, axis=1)
        data_oh.columns = self.binned_features
        return data_oh

    @staticmethod
    def accuracy_cost_bounds(min_costs):
        n = min_costs.shape[0]
        min_costs = min_costs[~np.isnan(min_costs)]
        min_costs = np.sort(min_costs)
        costs = np.zeros(min_costs.shape[0] + 1)
        corrects = np.zeros(min_costs.shape[0] + 1)
        # First element of each vector is 0
        for i in range(min_costs.shape[0]):
            costs[i + 1] = min_costs[: i + 1].mean()
            corrects[i + 1] = (i + 1) / n * 100
        return costs, corrects

    def lower_bounds(self, min_costs):
        n = min_costs.shape[0]
        min_costs = min_costs[min_costs != 0]
        min_costs = np.sort(min_costs)
        costs = np.zeros(min_costs.shape[0] + 1)
        corrects = np.zeros(min_costs.shape[0] + 1)
        for i in range(min_costs.shape[0]):
            corrects[i + 1] = (i + 1) / n * 100
            costs[i + 1] = min_costs[: i + 1].mean()
        return costs, corrects


class Apriori:
    # Takes x and (thresh OR affected_subgroup)
    # Compute features/widths/length/values from conditions
    def __init__(
        self,
        x,
        max_width=None,
        apriori_threshold=None,
        verbose=1,
        affected_subgroup=None,
        feature_values_tree=None,
    ):
        self.x = x
        self.max_width = max_width
        self.apriori_threshold = apriori_threshold
        self.affected_subgroup = affected_subgroup
        self.feature_values_tree = feature_values_tree
        if self.feature_values_tree is None:
            self.feature_values_tree = create_feature_values_tree(
                x.columns.values, use_values=True
            )

        if (self.apriori_threshold is not None) ^ (self.affected_subgroup is None):
            raise ValueError(
                "Please specify either an affected subgroup or an apriori threshold (and not both)"
            )

        if (self.affected_subgroup is not None) and (
            self.apriori_threshold is not None
        ):
            # Store indices of affected subgroup matches
            self.sub_idx = (self.x[affected_subgroup] == 1).values
            # Drop affected subgroup's feature so
            # apriori doesn't generate invalid rules
            self.x_drop = self.x.copy()
            for col in self.x.columns:
                if col.split()[0] == affected_subgroup.split()[0]:
                    self.x_drop = self.x_drop.drop(col, axis=1)
            self.conditions = pd.DataFrame(
                np.array([frozenset({affected_subgroup})]), columns=["itemsets"]
            )
        # as per paper, SD and RL are the same set generated by apriori
        elif self.apriori_threshold is not None:
            self.sub_idx = np.ones(self.x.shape[0], dtype=bool)
            # print(self.x, self.thresh, verbose, max_width)
            self.conditions = apriori(
                self.x,
                min_support=self.apriori_threshold,
                use_colnames=True,
                max_len=max_width,
                verbose=verbose,
            )

        self.values = self.conditions.itemsets.values
        self.features = self.compute_features(self.values)
        self.widths = self.conditions.apply(lambda i: len(i.itemsets), axis=1).values
        self.length = len(self.values)
        self.valid_idx = None

    def compute_features(self, values):
        features = np.zeros(len(values), dtype=object)
        for i in range(len(values)):
            features[i] = frozenset(map(self.feature_values_tree.get, values[i]))
        return features

    def reduce(self, utilise_bug=False, print_output=True):
        """
        IRREVERSIBLE
        """
        # INVESTIGATE why the broken np.unique implementation gives much better results on HELOC...
        # unfortunately np.unique fails with dtype=object, so we will manually implement a dictionary
        if print_output:
            print("Reducing Itemsets")
        n = self.length
        if n == 0:
            print("Size of itemset is 0 (no valid triples can be generated)")
            return
        if utilise_bug:  # useful bug (easter egg)
            un, co = np.unique(self.features, return_counts=True)
            valid = un[co > 1]
        else:
            unique_feature_counts = {}
            for feature in self.features:
                if feature in unique_feature_counts:
                    unique_feature_counts[feature] += 1
                else:
                    unique_feature_counts[feature] = 1
            valid = set()
            for feature in unique_feature_counts:
                if unique_feature_counts[feature] > 1:
                    valid.add(feature)

        self.valid_idx = np.zeros(self.length, dtype=bool)
        for i in range(self.length):
            if self.features[i] in valid:
                self.valid_idx[i] = True

        if self.valid_idx.any():
            self.conditions = self.conditions.loc[self.valid_idx]
            self.features = self.features[self.valid_idx]
            self.widths = self.widths[self.valid_idx]
            self.values = self.conditions.itemsets.values
            self.max_width = self.widths.max()
            self.length = len(self.values)
            if print_output:
                print("Size of Itemsets Reduced from {} to {}".format(n, self.length))
        else:
            self.conditions = None
            self.features = None
            self.widths = None
            self.values = None
            self.max_width = 0
            self.length = 0
            if print_output:
                print("Size of Itemsets Reduced from {} to {}".format(n, 0))
            print("No Valid Triples can be Generated from the Itemsets")


class TwoLevelRecourseSet:
    def __init__(self):
        # Class Attributes
        self.SD, self.RL = None, None
        self.triples = set()
        self.triples_array = None
        self.length = 0
        self.cfx_matrix = None  # counterfactuals after applying recourses to x_aff
        self.correct_matrix = None
        self.cost_matrix = None
        self.max_idxs = None
        self.cumulative_idxs = None
        self.correct_vector = None
        self.cost_vector = None
        self.accuracy = None
        self.average_cost = None
        self.correct_max = None
        self.correct_cumulative = None
        self.cover_matrix = None
        self.cover = None
        self.objectives = None
        self.ares = None
        self.featurecost = None
        self.featurechange = None

    def generate_triples(
        self, SD, RL, max_width, RL_reduction=False, then_generation=None
    ):
        """
        Stores SD, RL, triples and length
        """
        # Initialise SD and RL
        print("Computing Ground Set of Triples V")
        self.SD, self.RL = SD, RL
        # Apply RL-Reduction if required
        if RL_reduction is True:
            print("Reducing RL")
            n = self.RL.length
            self.RL.reduce(utilise_bug=False, print_output=False)
            print("RL Reduced from Size {} to {}".format(n, self.RL.length))
        if then_generation is not None:
            self.RL.features_tree = create_features_tree(self.RL.x.columns)
        disable_tqdm = False if self.SD.length > 1 else True
        for i in tqdm(range(self.SD.length), disable=disable_tqdm):
            for j in tqdm(range(self.RL.length), disable=(not disable_tqdm)):
                no_matching_features = self.SD.features[i].isdisjoint(
                    self.RL.features[j]
                )
                width_constraint = (self.SD.widths[i] + self.RL.widths[j]) <= max_width
                if width_constraint and no_matching_features:
                    width = self.RL.widths[j]
                    if then_generation is not None:
                        if then_generation == np.inf:
                            feature_values = (
                                []
                            )  # all feature values for conditions in RL[j]
                            for feature in self.RL.features[j]:
                                feature_values.append(self.RL.features_tree[feature])
                            RL2_values = np.array(
                                list(map(frozenset, itertools.product(*feature_values)))
                            )
                            RL2_features = self.RL.compute_features(RL2_values)
                        else:
                            # selects inputs that don't satisfy Outer-If/Inner-If conditions
                            # True if RL[j] not satisfied (row)
                            row = (self.RL.x[self.RL.values[j]] != 1).any(axis=1).values
                            # selects feature values at have features in Outer-If/Inner-If conditions
                            col = []
                            for feature in self.RL.features[j]:
                                # ALL feature values for each feature in each itemset
                                col += self.RL.features_tree[feature]
                            RL2 = Apriori(
                                x=self.RL.x[col][row],
                                max_width=width,
                                verbose=0,
                                apriori_threshold=then_generation,
                                feature_values_tree=self.RL.feature_values_tree,
                            )
                            RL2_values, RL2_features = RL2.values, RL2.features
                    else:
                        equal_width = self.RL.widths == width
                        RL2_values = self.RL.values[equal_width]
                        RL2_features = self.RL.features[equal_width]
                    if len(RL2_values) != 0:
                        for k in range(len(RL2_values)):
                            # checking widths is redundant since we check if sets have same features
                            identical_features = (
                                True
                                if then_generation == np.inf
                                else self.RL.features[j] == RL2_features[k]
                            )
                            identical_feature_values = (
                                self.RL.values[j] == RL2_values[k]
                            )
                            if identical_features and not identical_feature_values:
                                rule = (
                                    self.SD.values[i],
                                    self.RL.values[j],
                                    RL2_values[k],
                                )
                                self.triples.add(rule)
        self.length = len(self.triples)

    def evaluate_triples(
        self, ares, r=None, save_mode=0, disable_tqdm=False, plot_accuracy=False
    ):
        """
        Implement objective function parameter
        Method for evaluation of two level recourse sets. This needs a massive refactor with:
        self.evaluate, self.f_custom, self.f_ares, self.objective_terms
        (finds best correctness/cost and counterfactuals for each input)

        Inputs: R, final two level recourse set
                n_rules, maximum number of counterfactuals (that satisfied at least one rule)
                update_V: 0 is no rules are saved; 1 is all rules that satisfied at least
                one rule are saved; 2 is only rules that provided a lower cost counterfactual,
                or a successful counterfactual where one previously did not exist
                disable_tqdm/print_outputs: self-explanatory.
                correctness: doesn't compute costs.
        Outputs: vector of objective function values for each
                 member of X_aff (affected inputs requiring recourse)
                 vector of final counterfactuals for each member of X_aff
        """
        r = self.length if r is None else int(r)  # number of triples to evaluate
        if r > self.length:
            if len(self.triples_array) >= r:
                self.select_triples(r)
            else:
                raise ValueError(
                    "Number of elements for evaluation ({}) greater than number "
                    "of triples in set ({})".format(r, len(self.triples_array))
                )
        self.ares = ares  # originally sized input data
        n = self.ares.X_aff_original.shape[0]  # number of affected inputs
        self.correct_matrix = np.zeros((r, n), dtype=int)
        self.cost_matrix = np.zeros((r, n))
        self.cfx_matrix = np.zeros((r, *self.ares.X_aff_original.shape))
        self.triples_array = np.zeros(r, dtype=object)
        self.max_idxs = np.zeros(r, dtype=bool)
        self.cumulative_idxs = np.zeros(r, dtype=bool)
        cor_max, cor_cumulative = 0, np.zeros(n)
        self.correct_max = np.zeros(r)
        self.correct_cumulative = np.zeros((r, n))
        self.cover_matrix = np.zeros((r, n))
        self.featurecost = np.zeros(r)
        self.featurechange = np.zeros(r, dtype=int)

        for i, triple in tqdm(zip(range(r), self.triples), disable=disable_tqdm):
            (
                self.correct_matrix[i],
                self.cost_matrix[i],
                self.cfx_matrix[i],
                self.cover_matrix[i],
                self.featurecost[i],
                self.featurechange[i],
            ) = self.evaluate_triple(triple, ares)

            cor = self.correct_matrix[i].mean()
            if cor > cor_max:
                cor_max = cor
                self.max_idxs[i] = True
            self.correct_max[i] = cor_max

            self.correct_cumulative[i] = np.maximum(
                cor_cumulative, self.correct_matrix[i]
            )
            if self.correct_cumulative[i].sum() > cor_cumulative.sum():
                self.cumulative_idxs[i] = True
            cor_cumulative = self.correct_cumulative[i]

            self.triples_array[i] = triple
            i += 1
            if i == r:
                break

        if plot_accuracy:
            self.plot_accuracy()

        # Convert matrices to vectors
        if save_mode == 2:
            self.index_terms(self.cumulative_idxs)
        idx = self.correct_matrix == 0
        self.cost_matrix[idx] = np.nan
        if self.length != 0:
            self.correct_vector = self.correct_matrix.max(axis=0)
            self.accuracy = self.correct_vector.mean()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.cost_vector = np.nanmin(self.cost_matrix, axis=0)
                self.average_cost = np.nanmean(self.cost_vector)
            self.cover = self.cover_matrix.max(axis=0)

        if save_mode != 0:
            if save_mode == 1:
                self.triples = set(self.triples_array)
            elif save_mode == 2:
                self.triples = set(self.triples_array[self.cumulative_idxs])
            self.length = len(self.triples)

    def index_terms(self, idx):
        self.correct_matrix = self.correct_matrix[idx]
        self.cost_matrix = self.cost_matrix[idx]
        self.cover_matrix = self.cover_matrix[idx]
        self.featurechange = self.featurechange[idx]
        self.featurecost = self.featurecost[idx]

        if idx.any():
            self.cover = self.cover_matrix.max(axis=0)
            self.correct_vector = self.correct_matrix.max(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.cost_vector = np.nanmin(self.cost_matrix, axis=0)

        else:
            self.cover = np.zeros(idx.shape[0])
            self.correct_vector = np.zeros(idx.shape[0])
            self.cost_vector = np.zeros(idx.shape[0])

    def evaluate_triple(self, triple, ares):
        # Initialise
        n = ares.X_aff_original.shape[0]
        triple_cfx = ares.X_aff_original.copy()
        triple_corrects = np.zeros(n)
        triple_costs = np.zeros(n)

        # Compute (AReS requires all to be true)
        outer_ifs, inner_ifs, thens = list(triple[0]), list(triple[1]), list(triple[2])
        triple_cover = ares.X_aff[outer_ifs + inner_ifs].all(axis=1).values
        featurecost, featurechange, cfx = self.triple_cost(triple, ares, triple_cfx)

        if triple_cover.any():
            # Compute predictions and costs
            cfx = cfx[triple_cover]
            if ares.normalise:
                cfx_norm = (cfx.values - ares.means) / ares.stds
                corrects = ares.model.predict(cfx_norm)
            else:
                corrects = ares.model.predict(cfx.values)
            triple_corrects[triple_cover] = corrects * 100
            triple_costs[triple_cover] = featurechange
            triple_cfx[triple_cover] = cfx

        return (
            triple_corrects,
            triple_costs,
            triple_cfx,
            triple_cover,
            featurecost,
            featurechange,
        )

    @staticmethod
    def triple_cost(triple, ares, x_aff):
        outer_ifs, inner_ifs, thens = list(triple[0]), list(triple[1]), list(triple[2])
        featurecost, featurechange = 0, int(0)
        x_aff = x_aff.copy()
        # Generate counterfactuals to calculate predictions and costs
        # improve this by pairing directly the order of inner_ifs and thens
        # currently the features are not guaranteed to be in the same order
        # hence the search through inner_ifs (would also make reading triples easier)
        for then in thens:
            if (
                then not in triple[1]
            ):  # retain r[1] as opposed to inner-ifs for fast set retrieval
                # continuous variables
                then_feature = ares.feature_values_tree[then]
                for inner_if in inner_ifs:
                    if ares.feature_values_tree[inner_if] == then_feature:
                        if then in ares.binned_features_continuous:
                            then_idx = ares.features.index(
                                ares.feature_values_tree[then]
                            )
                            d = ares.bin_mids[then] - ares.bin_mids[inner_if]
                            featurechange += np.rint(
                                abs(d * ares.feature_costs_vector[then_idx])
                            )
                            x_aff[then_feature] += d
                        else:
                            then_idx = ares.features.index(then)
                            if then in ares.ordinal_features:
                                inner_if_idx = ares.features.index(inner_if)
                                inner_if_cost = ares.feature_costs_vector[inner_if_idx]
                                then_cost = ares.feature_costs_vector[then_idx]
                                featurechange += np.rint(abs(then_cost - inner_if_cost))
                            else:  # bulk write features since they all match conditions
                                featurechange += np.rint(1)  # categorical cost
                            x_aff[inner_if] = int(0)
                            x_aff[then] = int(1)
                        break  # match found, exit inner-if loop, resume then loop
        return featurecost, featurechange, x_aff  # featurecost not implemented

    def plot_accuracy(self, n_triples=None):
        if self.correct_max is not None:
            plt.figure(figsize=(4.5, 3), dpi=200)
            n_triples = len(self.correct_max) if n_triples is None else n_triples
            plt.plot(
                range(1, n_triples + 1),
                self.correct_cumulative.mean(axis=1)[:n_triples],
                label="All Selected Triples",
            )
            plt.plot(
                range(1, n_triples + 1),
                self.correct_max[:n_triples],
                label="Maximum Single Selected Triple",
            )
            plt.title(
                "Performance of Triples vs Number of Triples\nSelected "
                "in Ground Set of Total Length {}".format(self.length)
            )
            plt.ylabel("Recourse Accuracy (%)")
            plt.xlabel("Number of Triples in V Selected")
            plt.legend(loc="lower right")
            plt.show()
        else:
            raise ValueError(
                "Ground set must first be evaluated with self."
                "evaluate_triples() before accuracies can be plotted"
            )

    def select_triples(self, s=0, sort=True):
        """
        Optional method for selecting candidate rules with
        the highest individual objective function values
        Speeds up optimisation dramatically
        To be applied after self.counterfactuals()
        To be followed by self.optimise()

        Input: size of final candidate set, s
        """
        # Check for valid n_prefilter
        s = int(s)
        if s > len(self.triples_array):
            raise ValueError(
                "Number of filtered elements ({}) greater than length "
                "of recourse rules set ({})".format(s, len(self.triples_array))
            )
        # Defaults s=0 to whole set
        elif s == 0:
            s = len(self.triples_array)
        # Create set from filtered numpy array
        idx = np.argsort(self.objectives)[: -(s + 1) : -1] if sort else np.arange(s)
        self.triples = set()
        for i in self.triples_array[idx]:
            self.triples.add(i)
        self.length = len(self.triples)
        print("Candidate Set Filtered with Length:", s)
