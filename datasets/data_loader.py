# The code originates from https://github.com/danwley/GLOBE-CE/blob/main/datasets.py
# Paper for reference:
# Ley, Dan, Saumitra Mishra, and Daniele Magazzeni.
# "GLOBE-CE: a translation based approach for global counterfactual explanations.
# " International Conference on Machine Learning. PMLR, 2023.

"""

Contents:

========================== pep8 comment limit ==========================

split_dataset (function for generating train and test datasets)
dataset_loader (class for downloading and pre-processing datasets)

=============================== pep8 code limit ===============================

"""

import numpy as np
import os
from os import path
from sklearn import preprocessing
import pandas as pd
import urllib.request
import datetime


class dataset_loader:
    def __init__(
        self, name=None, data_path="datasets/", dropped_features=[], n_bins=None
    ):
        """
        Class for downloading and pre-processing datasets

        (optional arguments)
        name            : name of dataset (loads a specific dataset)
        data_path       : save location of dataset
                          (in the notebooks, this is w.r.t the root folder)
        dropped_features: list of feature names to remove
        n_bins          : number of equally-spaced bins for continuous features
                          (if None, then no binning)
        """

        # Dictionary of source URLs per dataset
        self.datasets = {
            "compas": "https://raw.githubusercontent.com/propublica/"
            + "compas-analysis/master/compas-scores-two-years.csv",
            "german_credit": "https://archive.ics.uci.edu/ml/machine-learning"
            + "-databases/statlog/german/german.data",
            "adult_income": "https://archive.ics.uci.edu/ml/machine-learning"
            + "-databases/adult/adult.data",
            "default_credit": "https://archive.ics.uci.edu/ml/machine-learning"
            + "databases/00350/default%20of%20credit%20card"
            + "%20clients.xls",
            "heloc": "https://drive.google.com/uc?id=1XnEgluPsPLN5It"
            + "OJ_DnoQnNxhtiDD8DE&export=download",
        }

        # Dictionary of features per dataset
        self.columns = {
            "compas": [
                "Sex",
                "Age_Cat",
                "Race",
                "C_Charge_Degree",
                "Priors_Count",
                "Time_Served",
                "Status",
            ],
            "german_credit": [
                "Existing-Account-Status",
                "Month-Duration",
                "Credit-History",
                "Purpose",
                "Credit-Amount",
                "Savings-Account",
                "Present-Employment",
                "Instalment-Rate",
                "Sex",
                "Guarantors",
                "Residence",
                "Property",
                "Age",
                "Installment",
                "Housing",
                "Existing-Credits",
                "Job",
                "Num-People",
                "Telephone",
                "Foreign-Worker",
                "Status",
            ],
            "adult_income": [
                "Age",
                "Workclass",
                "Fnlwgt",
                "Education",
                "Marital-Status",
                "Occupation",
                "Relationship",
                "Race",
                "Sex",
                "Capital-Gain",
                "Capital-Loss",
                "Hours-Per-Week",
                "Native-Country",
                "Status",
            ],
            "default_credit": [
                "Limit_Bal",
                "Sex",
                "Education",
                "Marriage",
                "Age",
                "Pay_0",
                "Pay_2",
                "Pay_3",
                "Pay_4",
                "Pay_5",
                "Pay_6",
                "Bill_Amt1",
                "Bill_Amt2",
                "Bill_Amt3",
                "Bill_Amt4",
                "Bill_Amt5",
                "Bill_Amt6",
                "Pay_Amt1",
                "Pay_Amt2",
                "Pay_Amt3",
                "Pay_Amt4",
                "Pay_Amt5",
                "Pay_Amt6",
                "Status",
            ],
            "heloc": [
                "ExternalRiskEstimate",
                "MSinceOldestTradeOpen",
                "MSinceMostRecentTradeOpen",
                "AverageMInFile",
                "NumSatisfactoryTrades",
                "NumTrades60Ever2DerogPubRec",
                "NumTrades90Ever2DerogPubRec",
                "PercentTradesNeverDelq",
                "MSinceMostRecentDelq",
                "MaxDelq2PublicRecLast12M",
                "MaxDelqEver",
                "NumTotalTrades",
                "NumTradesOpeninLast12M",
                "PercentInstallTrades",
                "MSinceMostRecentInqexcl7days",
                "NumInqLast6M",
                "NumInqLast6Mexcl7days",
                "NetFractionRevolvingBurden",
                "NetFractionInstallBurden",
                "NumRevolvingTradesWBalance",
                "NumInstallTradesWBalance",
                "NumBank2NatlTradesWHighUtilization",
                "PercentTradesWBalance",
                "Status",
            ],
        }

        # Dictionary of categorical features per dataset
        self.categorical_features = {
            "compas": ["Sex", "Age_Cat", "Race", "C_Charge_Degree"],
            "german_credit": [
                "Existing-Account-Status",
                "Credit-History",
                "Purpose",
                "Savings-Account",
                "Present-Employment",
                "Instalment-Rate",
                "Sex",
                "Guarantors",
                "Residence",
                "Property",
                "Installment",
                "Housing",
                "Existing-Credits",
                "Job",
                "Num-People",
                "Telephone",
                "Foreign-Worker",
            ],
            "adult_income": [
                "Workclass",
                "Education",
                "Marital-Status",
                "Occupation",
                "Relationship",
                "Race",
                "Sex",
                "Native-Country",
            ],
            "default_credit": [
                "Sex",
                "Education",
                "Marriage",
                "Pay_0",
                "Pay_2",
                "Pay_3",
                "Pay_4",
                "Pay_5",
                "Pay_6",
            ],
            "heloc": [],
        }

        # Dictionary of continuous features per dataset (computed)
        self.continuous_features = {}
        for dataset in self.columns:
            self.continuous_features[dataset] = []
            for column in self.columns[dataset][:-1]:
                if column not in self.categorical_features[dataset]:
                    self.continuous_features[dataset].append(column)

        # Initialization
        self.name = name
        if self.name is not None:  # process dataset if specified
            self.data_path = data_path
            self.n_bins = n_bins
            self.features = None  # processed in self.one_hot()
            self.features_tree = {}  # processed in self.one_hot()
            self.dropped_features = dropped_features
            self._load_dataset()  # download and process data
            if self.n_bins is not None:
                self.categorical_features[self.name] = list(self.features_tree.keys())
                self.continuous_features[self.name] = {}

    def _load_dataset(self):
        """
        Initialization method for preprocessing the data (one_hot encodings, feature names)
        """
        if self.name not in self.datasets:
            raise Exception("Dataset name does not match any known datasets.")
        if not path.exists(self.data_path):
            os.makedirs(self.data_path)

        url = self.datasets[self.name]
        file_name = "{}.data".format(self.name.split("_")[0])  # e.g. german.data
        file_address = self.data_path + file_name
        if not path.exists(file_address):
            print(
                "Downloading {} Dataset...".format(self.name.replace("_", " ").title())
            )
            urllib.request.urlretrieve(self.datasets[self.name], file_address)
            print("Dataset Successfully Downloaded.")

        if self.name == "compas":
            data = pd.read_csv(file_address)
            data = data.dropna(subset=["days_b_screening_arrest"])  # drop missing vals
            data = data.rename(columns={data.columns[-1]: "status"})
            data = self.process_compas(data)
            # Prepocess targets to Bad = 0, Good = 1
            cols = self.columns["compas"]
            data = data[[col.lower() for col in cols]]
            data.columns = cols
            data[data.columns[-1]] = 1 - data[data.columns[-1]]

        elif self.name == "german_credit":
            data = pd.read_csv(file_address, header=None, delim_whitespace=True)
            data.columns = self.columns[self.name]
            # Prepocess targets to Bad = 0, Good = 1
            data[data.columns[-1]] = 2 - data[data.columns[-1]]

        elif self.name == "adult_income":
            data = pd.read_csv(file_address, header=None, delim_whitespace=True)
            # remove redundant education num column (education processed in one_hot)
            data = data.drop(4, axis=1)
            # remove rows with missing values: '?,'
            data = data.replace("?,", np.nan)
            data = data.dropna()
            data.columns = self.columns[self.name]
            for col in data.columns[:-1]:
                # print(col)
                if col not in self.categorical_features[self.name]:
                    data[col] = data[col].apply(lambda x: float(x[:-1]))
                else:
                    data[col] = data[col].apply(lambda x: x[:-1])
            # Prepocess Targets to <=50K = 0, >50K = 1
            data[data.columns[-1]] = data[data.columns[-1]].replace(
                ["<=50K", ">50K"], [0, 1]
            )

        elif self.name == "default_credit":
            data = pd.read_excel(file_address, header=1)
            data = data.drop("ID", axis=1)
            data.columns = self.columns[self.name]
            data[data.columns[-1]] = 1 - data[data.columns[-1]]

        elif self.name == "heloc":
            data = pd.read_csv(file_address)
            # Remove rows where all NaN
            data = data[(data.iloc[:, 1:] >= 0).any(axis=1)]
            # Encode string labels
            data["RiskPerformance"] = data["RiskPerformance"].replace(
                ["Bad", "Good"], [0, 1]
            )
            # Move labels to final column (necessary for self.get_split)
            y = data.pop("RiskPerformance")
            data["RiskPerformance"] = y
            # Convert negative values to NaN
            data = data[data >= 0]
            # Replace NaN values with median
            nan_cols = data.isnull().any(axis=0)
            for col in data.columns:
                if nan_cols[col]:
                    data[col] = data[col].replace(np.nan, np.nanmedian(data[col]))

        else:
            raise Exception("Dataset name does not match any known datasets.")

        # Drop features and one hot encode
        for feature in self.dropped_features:
            data = data.drop(feature, axis=1)
        data_oh, self.features = self.one_hot(data)
        self.features.append(data.columns[-1])
        self.data = pd.concat([data_oh, data[data.columns[-1]]], axis=1)

    def one_hot(self, data):
        """
        Improvised method for one-hot encoding the data

        Input: data (whole dataset)
        Outputs: data_oh (one-hot encoded data)
                 features (list of feature values after one-hot encoding)
        """
        label_encoder = preprocessing.LabelEncoder()
        data_encode = data.copy()
        self.bins = {}
        self.bins_tree = {}

        # Assign encoded features to one hot columns
        data_oh, features = [], []
        for x in data.columns[:-1]:
            self.features_tree[x] = []
            categorical = x in self.categorical_features[self.name]
            if categorical:
                data_encode[x] = label_encoder.fit_transform(data_encode[x])
                cols = label_encoder.classes_
            elif self.n_bins is not None:
                data_encode[x] = pd.cut(
                    data_encode[x].apply(lambda x: float(x)), bins=self.n_bins
                )
                cols = data_encode[x].cat.categories
                self.bins_tree[x] = {}
            else:
                data_oh.append(data[x])
                features.append(x)
                continue

            one_hot = pd.get_dummies(data_encode[x])
            if self.name == "compas" and x.lower() == "age_cat":
                one_hot = one_hot[[2, 0, 1]]
                cols = cols[[2, 0, 1]]
            data_oh.append(one_hot)
            for col in cols:
                feature_value = x + " = " + str(col)
                features.append(feature_value)
                self.features_tree[x].append(feature_value)
                if not categorical:
                    self.bins[feature_value] = col.mid
                    self.bins_tree[x][feature_value] = col.mid

        data_oh = pd.concat(data_oh, axis=1, ignore_index=True)
        data_oh.columns = features
        return data_oh, features

    def get_split(
        self,
        ratio=0.8,
        normalise=True,
        shuffle=False,
        return_mean_std=False,
        print_outputs=False,
    ):
        """
        Method for returning training/test split with optional normalisation/shuffling

        Inputs: ratio (proportion of training data)
                normalise (if True, normalises data)
                shuffle (if True, shuffles data)
                return_mean_std (if True, returns mean and std. dev. of training data)
        Outputs: train and test data
        """
        if shuffle:
            self.data = self.data.sample(frac=1)
        data = self.data.values
        train_idx = int(data.shape[0] * ratio)
        x_train, y_train = data[:train_idx, :-1], data[:train_idx, -1]
        x_test, y_test = data[train_idx:, :-1], data[train_idx:, -1]

        if print_outputs:
            print(
                "\033[1mProportion of 1s in Training Data:\033[0m {}%".format(
                    round(np.average(y_train) * 100, 2)
                )
            )
            print(
                "\033[1mProportion of 1s in Test Data:\033[0m {}%".format(
                    round(np.average(y_test) * 100, 2)
                )
            )

        x_means, x_stds = x_train.mean(axis=0), x_train.std(axis=0)

        if normalise:
            x_train = (x_train - x_means) / x_stds
            x_test = (x_test - x_means) / x_stds

        if return_mean_std:
            return x_train, y_train, x_test, y_test, x_means, x_stds
        return x_train, y_train, x_test, y_test

    def process_compas(self, data):
        """
        Additional method to process specifically the COMPAS dataset

        Input: data (whole dataset)
        Output: data (whole dataset)
        """
        data = data.to_dict("list")
        for k in data.keys():
            data[k] = np.array(data[k])

        dates_in = data["c_jail_in"]
        dates_out = data["c_jail_out"]
        # this measures time in Jail
        time_served = []
        for i in range(len(dates_in)):
            di = datetime.datetime.strptime(dates_in[i], "%Y-%m-%d %H:%M:%S")
            do = datetime.datetime.strptime(dates_out[i], "%Y-%m-%d %H:%M:%S")
            time_served.append((do - di).days)
        time_served = np.array(time_served)
        time_served[time_served < 0] = 0
        data["time_served"] = time_served

        """ Filtering the data """
        # These filters are as taken by propublica
        # (refer to https://github.com/propublica/compas-analysis)
        # If the charge date of a defendants Compas scored crime was not within 30 days
        # from when the person was arrested, we assume that because of data quality
        # reasons, that we do not have the right offense.
        idx = np.logical_and(
            data["days_b_screening_arrest"] <= 30,
            data["days_b_screening_arrest"] >= -30,
        )

        # We coded the recidivist flag -- is_recid -- to be -1
        # if we could not find a compas case at all.
        idx = np.logical_and(idx, data["is_recid"] != -1)

        # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of
        # 'O' -- will not result in Jail time are removed (only two of them).
        idx = np.logical_and(idx, data["c_charge_degree"] != "O")
        # F: felony, M: misconduct

        # We filtered the underlying data from Broward county to include only those rows
        # representing people who had either recidivated in two years, or had at least two
        # years outside of a correctional facility.
        idx = np.logical_and(idx, data["score_text"] != "NA")

        # select the examples that satisfy this criteria
        for k in data.keys():
            data[k] = data[k][idx]
        return pd.DataFrame(data)