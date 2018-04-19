from baselines import logger
import baselines.common as common
import numpy as np
from sklearn.externals import joblib
from dci import DCI
import tensorflow as tf
import time

class LinearDensityValueFunction(object):

    def __init__(self, n_neighbors, ob_dim, ac_dim):
        self.n_neighbors = n_neighbors
        self.dim = 2 * ob_dim + 2 * ac_dim + 1
        self.X, self.y = None, None
        self.timestep_to_numdatapoints_dict = {}

        # prioritized DCI
        self.num_comp_indices = 2
        self.num_simp_indices = 7
        self.num_levels = 2
        self.construction_field_of_view = 10
        self.construction_prop_to_retrieve = 0.002
        self.query_field_of_view = 100
        self.query_prop_to_retrieve = 0.05

        self.dci = DCI(self.dim, self.num_comp_indices, self.num_simp_indices)

        # define the graph for linear regression
        self.X_linreg = tf.placeholder(tf.float32, shape=[None, self.n_neighbors, self.dim])
        self.y_linreg = tf.placeholder(tf.float32, shape=[None, self.n_neighbors]) 
        self.X_query_linreg = tf.placeholder(tf.float32, shape=[None, self.dim])
        self.ridge = tf.eye(self.dim) * 1e-2

        Xt_linreg = tf.transpose(self.X_linreg, [0, 2, 1])
        self.inv = tf.matrix_inverse(tf.matmul(Xt_linreg, self.X_linreg) + self.ridge)
        end = tf.matmul(Xt_linreg, tf.expand_dims(self.y_linreg, -1))
        self.weights = tf.squeeze(tf.matmul(self.inv, end), -1)
        self.y_pred = tf.reduce_sum(self.X_query_linreg * self.weights, -1)

    def _preproc(self, path):
        l = path["reward"].shape[0]
        act = path["action_dist"].astype('float32')
        al = np.arange(l).reshape(-1,1) / 10.0
        X = np.concatenate([path['observation'], act, al], axis=1)
        return X

    def predict(self, path):
        X_query = self._preproc(path)
        if not self.is_fit():
            return np.random.rand(X_query.shape[0])
        return self._predict_linreg(X_query)

    def _predict_linreg(self, X_query):
        nn_idx, _ = dci_db.query(X_query, num_neighbours=self.n_neighbors, field_of_view=self.query_field_of_view, prop_to_retrieve=self.query_prop_to_retrieve, blind=True)
        X, y = self.X[nn_idx], self.y[nn_idx]
        # X_t = X.swapaxes(1, 2)
        # inv = np.linalg.inv(X_t @ X + 1e-2 * np.eye(X.shape[2]))
        # end = np.einsum('ijk,ik->ij', X_t, y)
        # weights = np.einsum('ijk,ik->ij', inv, end)
        # y_pred = (X_query * weights).sum(axis=1)
        y_pred = tf.get_default_session().run(self.y_pred, 
            feed_dict={self.X_linreg: X, self.y_linreg: y, self.X_query_linreg: X_query})
        return y_pred

    def fit(self, paths, targvals, timesteps_so_far):
        X = np.concatenate([self._preproc(p) for p in paths])
        y = np.concatenate(targvals)

        if self.is_fit():
            logger.record_tabular("EVBefore", common.explained_variance(self._predict_linreg(X), y))

        if self.X is None or self.y is None:
            self.X = np.array(X)
            self.y = np.array(y)
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.hstack((self.y, y))

        self.dci.add(self.X, num_levels=self.num_levels, 
            field_of_view=self.construction_field_of_view, prop_to_retrieve=self.construction_prop_to_retrieve)

        self.timestep_to_numdatapoints_dict[timesteps_so_far] = self.X.shape[0]

        if self.is_fit():
            logger.record_tabular("EVAfter", common.explained_variance(self._predict_linreg(X), y))

    def is_fit(self):
        return self.X.shape[0] > self.n_neighbors

    def save_model(self, data_filename, dict_filename):
        joblib.dump(self.X, data_filename)
        joblib.dump(self.timestep_to_numdatapoints_dict, dict_filename)