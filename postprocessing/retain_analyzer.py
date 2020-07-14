import matplotlib.pyplot as plt
import os
import numpy as np
import misc
import misc.constants as cs
import pickle
from pathlib import Path
from preprocessing.preprocessing import preprocessing
from processing.models.retain_atl import RETAIN_ATL
from misc.utils import locate_params


class RetainAnalyzer():
    def __init__(self, dataset, ph, hist, experiment, params):
        self.dataset = dataset
        self.ph = ph // cs.freq
        self.hist = hist // cs.freq
        self.exp = experiment
        self.params = locate_params(params)
        self.train, self.valid, self.test, self.scalers = {}, {}, {}, {}

    def _load_subject_data(self, subject):
        if not subject in list(self.train.keys()):
            train_sbj, valid_sbj, test_sbj, scalers_sbj = preprocessing(self.dataset, subject, self.ph,
                                                                        self.hist, cs.day_len_f)
            self.train[subject] = train_sbj
            self.valid[subject] = valid_sbj
            self.test[subject] = test_sbj
            self.scalers[subject] = scalers_sbj

    def _load_all_subjects_data(self):
        for subject in misc.datasets.datasets[self.dataset]["subjects"]:
            self._load_all_subjects_data(subject)

    def _create_models(self, subject):
        models = []
        for train_i, valid_i, test_i in zip(self.train[subject], self.valid[subject], self.test[subject]):
            model = RETAIN_ATL(subject, self.ph, self.params, train_i, valid_i, test_i)
            file_name = "RETAIN_ATL_" + self.dataset + subject + ".pt"
            file_path = os.path.join(cs.path, "processing", "models", "weights", self.exp, file_name)
            model.load(file_path)
            models.append(model)

        return models

    def compute_contribution_subject(self, subject, eval_set="test"):
        self._load_subject_data(subject)

        models = self._create_models(subject)

        contrib_an = []
        for model in models:
            contrib_an.append(model.contribution_an(eval_set))

        return contrib_an

    def compute_max_contrib(self, subject, eval_set):
        contrib_an = self.compute_contribution_subject(subject, eval_set)
        max_contrib = np.max(contrib_an, axis=1)
        max_contrib = np.flip(max_contrib, axis=2)
        mean_max_contrib, std_max_contrib = np.mean(max_contrib, axis=0), np.std(max_contrib, axis=0)
        return mean_max_contrib, std_max_contrib

    def compute_mean_std_max_contrib(self, subject, eval_set="test"):
        if subject == "all":
            max_contrib = []
            for sbj in misc.datasets.datasets[self.dataset]["subjects"]:
                mean_max_contrib_sbj, _ = self.compute_max_contrib(sbj, eval_set)
                max_contrib.append(mean_max_contrib_sbj)
            mean_max_contrib, std_max_contrib = np.mean(max_contrib, axis=0), np.std(max_contrib, axis=0)
        else:
            mean_max_contrib, std_max_contrib = self.compute_max_contrib(subject, eval_set)

        return mean_max_contrib, std_max_contrib

    def compute_stimuli_indexes_subject(self, stimuli, subject, eval_set="test", max_lag=5):
        self._load_subject_data(subject)

        if eval_set == "train":
            data = self.train[subject]
        elif eval_set == "valid":
            data = self.valid[subject]
        elif eval_set == "test":
            data = self.test[subject]

        last_idx = self.params["hist"] // cs.freq - 1

        stimuli_col_idx = np.where(data[0].columns == stimuli + "_" + str(last_idx))[0][0]
        stimuli_data = [data_split.loc[:, stimuli + "_" + str(last_idx)] for data_split in data]
        stimuli_data = [
            stimuli_data_split * scalers_split.scale_[stimuli_col_idx - 1] + scalers_split.mean_[stimuli_col_idx - 1]
            for stimuli_data_split, scalers_split in zip(stimuli_data, self.scalers[subject])]

        non_zero_stimuli_idx = [np.where(~np.isclose(stimuli_data_split, 0))[0] for stimuli_data_split in stimuli_data]
        toofar_idx = [_ + max_lag > len(data[0]) for _ in non_zero_stimuli_idx]
        non_zero_stimuli_idx = [_[~toofar_idx_split] for _, toofar_idx_split in zip(non_zero_stimuli_idx, toofar_idx)]
        return non_zero_stimuli_idx

    def compute_contrib_after_stimuli_subject(self, stimuli, subject, eval_set="test", lag=0):
        stimuli_idx = self.compute_stimuli_indexes_subject(stimuli, subject, eval_set)
        contrib = self.compute_contribution_subject(subject, eval_set)
        contrib_after_stimuli = [contrib_split[stimuli_idx_split + lag] for stimuli_idx_split, contrib_split in
                                 zip(stimuli_idx, contrib)]
        return np.mean(np.mean(contrib_after_stimuli, axis=0), axis=0), np.mean(np.std(contrib_after_stimuli, axis=0),
                                                                                axis=0)

    def compute_mean_std_contrib_after_stimuli(self, stimuli, subject, eval_set="test", lag=0):
        if subject == "all":
            max_contrib = []
            for sbj in misc.datasets.datasets[self.dataset]["subjects"]:
                mean_max_contrib_sbj, _ = self.compute_contrib_after_stimuli_subject(stimuli, sbj, eval_set, lag)
                max_contrib.append(mean_max_contrib_sbj)
            mean_max_contrib, std_max_contrib = np.mean(max_contrib, axis=0), np.std(max_contrib, axis=0)
        else:
            mean_max_contrib, std_max_contrib = self.compute_contrib_after_stimuli_subject(stimuli, subject, eval_set, lag)
        return mean_max_contrib, std_max_contrib

    def plot_evolution_after_stimuli(self, stimuli, max_lag=5, subject="all", eval_set="test",history_limit=40):
        mean_contrib_after_stimuli, std_contrib_after_stimuli = [], []
        for lag in range(max_lag + 1):
            mean_contrib_after_stimuli_lag, std_contrib_after_stimuli_lag = self.compute_mean_std_contrib_after_stimuli(
                stimuli, subject, eval_set, lag)
            mean_contrib_after_stimuli.append(mean_contrib_after_stimuli_lag)
            std_contrib_after_stimuli.append(std_contrib_after_stimuli_lag)



        fig, axes = plt.subplots(ncols=max_lag+1, nrows=1, figsize=(21, 5))
        history_limit_f = history_limit // cs.freq
        time = np.arange(0, self.params["hist"], cs.freq)[:history_limit_f]

        for i, (mean, std) in enumerate(zip(mean_contrib_after_stimuli,std_contrib_after_stimuli)):
            mean, std = np.flip(mean,axis=0), np.flip(std,axis=0)

            axes[i].plot(time, mean[:history_limit_f, 0], color="blue", label="glycemia")
            axes[i].fill_between(time, mean[:history_limit_f, 0] - std[:history_limit_f, 0],
                                 mean[:history_limit_f, 0] + std[:history_limit_f, 0], alpha=0.2, edgecolor='blue',
                             facecolor="blue")

            axes[i].plot(time, mean[:history_limit_f, 2], color="green", label="insulin")
            axes[i].fill_between(time, mean[:history_limit_f, 2] - std[:history_limit_f, 2],
                                 mean[:history_limit_f, 2] + std[:history_limit_f, 2], alpha=0.2, edgecolor='green',
                             facecolor="green")

            axes[i].plot(time, mean[:history_limit_f, 1], color="red", label="CHO")
            axes[i].fill_between(time, mean[:history_limit_f, 1] - std[:history_limit_f, 1],
                                 mean[:history_limit_f, 1] + std[:history_limit_f, 1], alpha=0.2, edgecolor='red',
                             facecolor="red")

    def plot_max_contribution(self, subject="all", eval_set="test"):
        mean_max_contrib, std_max_contrib = self.compute_mean_std_max_contrib(subject, eval_set)

        time = np.arange(0, self.params["hist"], cs.freq)

        plt.figure()
        plt.plot(time, mean_max_contrib[:, 0], color="blue", label="glycemia")
        plt.fill_between(time, mean_max_contrib[:, 0] - std_max_contrib[:, 0],
                         mean_max_contrib[:, 0] + std_max_contrib[:, 0], alpha=0.5, edgecolor='blue', facecolor="blue")

        plt.plot(time, mean_max_contrib[:, 2], color="green", label="insulin")
        plt.fill_between(time, mean_max_contrib[:, 2] - std_max_contrib[:, 2],
                         mean_max_contrib[:, 2] + std_max_contrib[:, 2], alpha=0.5, edgecolor='green',
                         facecolor="green")

        plt.plot(time, mean_max_contrib[:, 1], color="red", label="CHO")
        plt.fill_between(time, mean_max_contrib[:, 1] - std_max_contrib[:, 1],
                         mean_max_contrib[:, 1] + std_max_contrib[:, 1], alpha=0.5, edgecolor='red', facecolor="red")

        plt.xlabel("History [min]")
        plt.ylabel("Maximum absolute normalized contribution")
        plt.legend()
        plt.title("Maximum absolute normalized contribution for dataset " + self.dataset + " and subject " + subject)

# class RetainWeights():
#     def __init__(self, dataset, subject, ph, experiment, weights=None):
#         self.experiment = experiment
#         self.ph = ph
#         self.dataset = dataset
#         self.subject = subject
#         self.freq = np.max([cs.freq, misc.datasets.datasets[dataset]["glucose_freq"]])
#
#         if weights is None:
#             self.params, self.results = self.load_weights()
#         else:
#             self.weights = weights
#
#     def load_weights(self):
#         file = self.dataset + "_" + self.subject + ".npy"
#         path = os.path.join(cs.path, "results", "retain_weights", self.experiment, file)
#
#         with open(path, 'rb') as handle:
#             weights = pickle.load(handle)
#
#         return weights
#
#     def save_weights(self):
#         file = self.dataset + "_" + self.subject + ".npy"
#         dir = os.path.join(cs.path, "results", "retain_weights", self.experiment)
#         path = os.path.join(dir, file)
#         Path(dir).mkdir(parents=True, exist_ok=True)
#
#         with open(path, 'wb') as handle:
#             pickle.dump(self.weights, handle)
