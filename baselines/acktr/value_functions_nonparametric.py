from baselines import logger
import baselines.common as common
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

class LinearDensityValueFunction(object):

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.X, self.y = [], []
        self.is_fit = False

    def _preproc(self, path):
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1)/10.0
        act = path["action_dist"].astype('float32')
        X = np.concatenate([path['observation'], act, al, np.ones((l, 1))], axis=1)
        return X

    def predict(self, path):
        X = self._preproc(path)
        if self.is_fit:
            return self.neigh.predict(X)
        return np.random.rand(X.shape[0])

    def fit(self, paths, targvals):
        X = np.concatenate([self._preproc(p) for p in paths])
        y = np.concatenate(targvals)
        if self.is_fit:
            logger.record_tabular("EVBefore", common.explained_variance(self.neigh.predict(X), y))
        self.X.extend(X)
        self.y.extend(y)
        if len(X) > self.n_neighbors:
            self.neigh.fit(X, y)
            self.is_fit = True
        if self.is_fit:
            logger.record_tabular("EVAfter", common.explained_variance(self.neigh.predict(X), y))

    def save_model(self, filename):
        joblib.dump(self.neigh, filename)

    def load_model(self, filename):
        self.neigh = joblib.load(filename)
        self.is_fit = True

def pathlength(path):
    return path["reward"].shape[0]
