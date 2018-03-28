from baselines import logger
import baselines.common as common
import numpy as np

from sklearn.neighbors import KDTree, KNeighborsRegressor
from sklearn.externals import joblib

class LinearDensityValueFunction(object):

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.is_fit, self.has_data = False, False
        self.neigh = KNeighborsRegressor(n_neighbors=n_neighbors)

    def _preproc(self, path):
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1)/10.0
        act = path["action_dist"].astype('float32')
        X = np.concatenate([path['observation'], act, al, np.ones((l, 1))], axis=1)
        return X

    def predict(self, path, linreg=False, knn=False):
        X_query = self._preproc(path)
        # print('printing features of X before deletion')
        # for i in range(X_query.shape[1]):
        #     print(str(i), X_query[:, i])
        X_query = np.delete(X_query, [4, 5, 10, 15, 16, 21, 24, 25], axis=1)
        # print('printing features of X after deletion')
        # for i in range(X_query.shape[1]):
        #     print(str(i), X_query[:, i])
        if not self.is_fit:
            return np.random.rand(X_query.shape[0])
        if not linreg and not knn:
            if np.random.rand() < 0.5:
                linreg = True
            else:
                knn = True
        if linreg:
            return self._predict_linreg(X_query)
        if knn:
            return self.neigh.predict(X_query)

    def _predict_linreg(self, X_query):
        ind = self.tree.query(X_query, k=self.n_neighbors, return_distance=False)
        X, y = self.X[ind], self.y[ind]
        # X = np.insert(X, X.shape[2], 1, axis=2)
        # X_query = np.insert(X_query, X_query.shape[1], 1, axis=1)

        X_t = X.swapaxes(1, 2)
        # print(np.linalg.svd(X_t @ X, compute_uv=False)[0])
        inv = np.linalg.inv(X_t @ X) # + 1e-10 * np.eye(X.shape[2])
        end = np.einsum('ijk,ik->ij', X_t, y)
        weights = np.einsum('ijk,ik->ij', inv, end)
        y_pred = (X_query * weights).sum(axis=1)
        return y_pred

    def fit(self, paths, targvals):
        X = np.concatenate([self._preproc(p) for p in paths])
        y = np.concatenate(targvals)
        X = np.delete(X, [4, 5, 10, 15, 16, 21, 24, 25], axis=1)
        if not self.has_data:
            self.X = np.array(X)
            self.y = np.array(y)
            self.has_data = True
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.hstack((self.y, y))
        self.tree = KDTree(self.X, leaf_size=10)
        self.neigh.fit(self.X, self.y)
        self.is_fit = len(self.X) > self.n_neighbors
        # if self.is_fit:
        #     logger.record_tabular("EVBefore", common.explained_variance(self._predict(X), y))
        # if self.is_fit:
        #     logger.record_tabular("EVAfter", common.explained_variance(self._predict(X), y))

    def save_model(self, filename_linreg, filename_knn):
        joblib.dump(self.neigh, filename_knn)
        joblib.dump(self.tree, filename_linreg)

    def load_model(self, filename_linreg, filename_knn):
        self.neigh = joblib.load(filename_knn)
        self.tree = joblib.load(filename_linreg)
        self.is_fit = True

def pathlength(path):
    return path["reward"].shape[0]
