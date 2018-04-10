from baselines import logger
import baselines.common as common
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.externals import joblib

class LinearDensityValueFunction(object):

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.is_fit, self.has_data = False, False
        self.delete_features = []

    def predict(self, X_query):
        X_query = X_query.reshape((X_query.shape[0], -1))
        X_query = np.delete(X_query, self.delete_features, axis=1)
        if not self.is_fit:
            return np.random.rand(X_query.shape[0])
        return self._predict_linreg(X_query)

    def _predict_linreg(self, X_query):
        ind = self.tree.query(X_query, k=self.n_neighbors, return_distance=False)
        X, y = self.X[ind], self.y[ind]
        X_t = X.swapaxes(1, 2)
        # for i in range(X.shape[2]):
        #     # print(X[0, :, i])
        #     if np.all(X[0, :, i] == 0):
        #         print('{} has all 0'.format(i))
        inv = np.linalg.inv(X_t @ X) # + 1e-10 * np.eye(X.shape[2])
        end = np.einsum('ijk,ik->ij', X_t, y)
        weights = np.einsum('ijk,ik->ij', inv, end)
        y_pred = (X_query * weights).sum(axis=1)
        return y_pred

    def fit(self, X, y):
        # ## filter out the first obs since its (obs, 0) state
        # X = np.concatenate([self._preproc(p)[1:] for p in paths])
        # y = np.concatenate([targval[1:] for targval in targvals])
        X = X.reshape((X.shape[0], -1))
        X = np.delete(X, self.delete_features, axis=1)
        if not self.has_data:
            self.X = np.array(X)
            self.y = np.array(y)
            self.has_data = True
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.hstack((self.y, y))
        self.tree = KDTree(self.X, leaf_size=10)
        self.is_fit = len(self.X) > self.n_neighbors
        # if self.is_fit:
        #     logger.record_tabular("EVBefore", common.explained_variance(self._predict(X), y))
        # if self.is_fit:
        #     logger.record_tabular("EVAfter", common.explained_variance(self._predict(X), y))

    def save_model(self, filename_linreg=None, filename_knn=None):
        if filename_knn:
            joblib.dump(self.neigh, filename_knn)
        if filename_linreg:
            joblib.dump(self.tree, filename_linreg)

    def load_model(self, filename_linreg=None, filename_knn=None):
        if filename_knn:
            self.neigh = joblib.load(filename_knn)
        if filename_linreg:
            self.tree = joblib.load(filename_linreg)
        self.is_fit = True

def pathlength(path):
    return path["reward"].shape[0]
