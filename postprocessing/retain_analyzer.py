import os
import numpy as np
import misc
import misc.constants as cs
import pickle
from pathlib import Path

class RetainWeights():
    def __init__(self, dataset, subject, ph, experiment, weights=None):
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.subject = subject
        self.freq = np.max([cs.freq, misc.datasets.datasets[dataset]["glucose_freq"]])

        if weights is None:
            self.params, self.results = self.load_weights()
        else:
            self.weights = weights

    def load_weights(self):
        file = self.dataset + "_" + self.subject + ".npy"
        path = os.path.join(cs.path, "results", "retain_weights", self.experiment, file)

        with open(path, 'rb') as handle:
            weights = pickle.load(handle)

        return weights

    def save_weights(self):
        file = self.dataset + "_" + self.subject + ".npy"
        dir = os.path.join(cs.path, "results", "retain_weights", self.experiment)
        path = os.path.join(dir, file)
        Path(dir).mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as handle:
            pickle.dump(self.weights, handle)

