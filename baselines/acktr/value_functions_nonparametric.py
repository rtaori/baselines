from baselines import logger
import baselines.common as common
import numpy as np

from sklearn.neighbors import KDTree, KNeighborsRegressor
from sklearn.externals import joblib

class LinearDensityValueFunction(object):

    def __init__(self, n_neighbors, linreg=True, knn=True):
        self.n_neighbors = n_neighbors
        self.is_fit, self.has_data = False, False
        self.linreg, self.knn = linreg, knn
        if self.knn:
            self.neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        # self.delete_features = [10, 21, 24, 25] #reacher
        self.delete_features = [25, 26, 27]

    def _preproc(self, path):
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1)/10.0
        act = path["action_dist"].astype('float32')
        X = np.concatenate([path['observation'], act, al, np.ones((l, 1))], axis=1)
        return X

    def predict(self, path, linreg=True, knn=False):
        X_query = self._preproc(path)
        X_query = np.delete(X_query, self.delete_features, axis=1) #only for reacher
        if not self.is_fit:
            return np.random.rand(X_query.shape[0])
        if linreg:
            return self._predict_linreg(X_query)
        if knn:
            return self.neigh.predict(X_query)

    def _predict_linreg(self, X_query):
        ind = self.tree.query(X_query, k=self.n_neighbors, return_distance=False)
        X, y = self.X[ind], self.y[ind]
        X_t = X.swapaxes(1, 2)
        # for i in range(X.shape[2]):
        #     # print(X[0, :, i])
        #     if np.all(X[0, :, i] == 1):
        #         print('{} has all 1'.format(i))
        inv = np.linalg.inv(X_t @ X) # + 1e-10 * np.eye(X.shape[2])
        end = np.einsum('ijk,ik->ij', X_t, y)
        weights = np.einsum('ijk,ik->ij', inv, end)
        y_pred = (X_query * weights).sum(axis=1)
        return y_pred

    def fit(self, paths, targvals):
        X = np.concatenate([self._preproc(p) for p in paths])
        y = np.concatenate(targvals)
        X = np.delete(X, self.delete_features, axis=1) #only for reacher
        if not self.has_data:
            self.X = np.array(X)
            self.y = np.array(y)
            self.has_data = True
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.hstack((self.y, y))
        if self.linreg:
            self.tree = KDTree(self.X, leaf_size=10)
        if self.knn:
            self.neigh.fit(self.X, self.y)
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
