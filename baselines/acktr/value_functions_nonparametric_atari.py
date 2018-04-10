from baselines import logger
import baselines.common as common
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.externals import joblib
from sklearn.decomposition import PCA

import time

import tensorflow as tf
import torch

class LinearDensityValueFunction(object):

    def __init__(self, n_neighbors, sess):
        self.n_neighbors = n_neighbors
        self.sess = sess
        self.is_fit, self.has_data = False, False
        self.delete_features = []
        self.pca = PCA(n_components=10)

        self.matrix = tf.placeholder(tf.float32, shape=[None, None, None])
        self.rhs = tf.placeholder(tf.float32, shape=[None, None])
        self.query = tf.placeholder(tf.float32, shape=[None, None])
        rhs = tf.expand_dims(self.rhs, -1)
        weights = tf.matrix_solve_ls(self.matrix, rhs)
        self.preds = tf.reduce_sum(self.query * tf.squeeze(weights, -1), -1)


    def predict(self, X_query):
        X_query = X_query.reshape((X_query.shape[0], -1))
        X_query = np.delete(X_query, self.delete_features, axis=1)
        if not self.is_fit:
            return np.random.rand(X_query.shape[0])
        return self._predict_linreg(X_query)

    def _predict_linreg(self, X_query):
        ind = self.tree.query(X_query, k=self.n_neighbors, return_distance=False)
        X, y = self.X[ind], self.y[ind]
        start = time.time()

        X, y, X_query = torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(X_query).float()
        X_flat = X.view(-1, X.shape[2])
        X_flat_mean = X_flat.mean(-1, True)
        X_flat = X_flat - X_flat_mean
        U, S, V = torch.svd(X_flat)
        X_pca = torch.mm(X_flat, V[:10])
        X_query_pca = torch.mm(X_query - X_flat_mean, V[:10])
        X_pca, X_query_pca = X.view(X.shape[0], X.shape[1], -1), X_query.view(X_query.shape[0], X_query.shape[1], -1)
        X_pca, X_query_pca = np.array(X_pca), np.array(X_query_pca)
        # X_pca = self.pca.fit_transform(X.reshape((-1, X.shape[2]))).reshape((X.shape[0], X.shape[1], -1))
        # X_query_pca = self.pca.transform(X_query)
        # print('pca', time.time() - start)
        # print('all zeros', not X.any())
        # for query in X:
        #     print(query)
        # print(X[0, :, :100])
        # end = np.matmul(X_t, np.expand_dims(y, 2))
        # end = np.einsum('ijk,ik->ij', X_t, y)
        # weights = np.linalg.solve(X_t @ X, end)
        # weights = np.einsum('ijk,ik->ij', inv, end)
        # y_pred = (X_query * weights).sum(axis=1)
        y_pred = self.sess.run(self.preds, feed_dict={self.matrix: X_pca, self.rhs: y, self.query: X_query_pca})
        print('pca + ls', time.time() - start)
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