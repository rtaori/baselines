from baselines import logger
import baselines.common as common
import numpy as np
from sklearn.externals import joblib
from dci import DCI
import tensorflow as tf
import time
from baselines.acktr.numpy_deque import NumpyDeque

# This is the class that handles density-based critic for mujoco envs
class LinRegVF(object):

    def __init__(self, n_neighbors, timestep_window, ob_dim, ac_dim):
        self.n_neighbors = n_neighbors
        self.dim = 2 * ob_dim + 2 * ac_dim + 1

        self.X_db = NumpyDeque(max_capacity=timestep_window)
        self.y_db = NumpyDeque(max_capacity=timestep_window, one_dimensional=True)

        # prioritized DCI parameters
        self.num_comp_indices = 2
        self.num_simp_indices = 7
        self.num_levels = 2
        self.construction_field_of_view = 10
        self.construction_prop_to_retrieve = 0.002
        self.query_field_of_view = 100
        self.query_prop_to_retrieve = 0.05

        # define the graph for linear regression
        self.X_linreg = tf.placeholder(tf.float32, shape=[None, self.n_neighbors, self.dim])
        self.y_linreg = tf.placeholder(tf.float32, shape=[None, self.n_neighbors]) 
        self.X_query_linreg = tf.placeholder(tf.float32, shape=[None, self.dim])
        self.ridge = tf.eye(self.dim) * 1e-2

        # actual graph for linear regression
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

    # call this function to get predictions
    def predict(self, path):
        X_query = self._preproc(path)
        if not self.is_fit():
            return np.random.rand(X_query.shape[0])
        return self._predict_linreg(X_query)

    # this specifically uses linreg to do the prediction
    def _predict_linreg(self, X_query):
        # query from the DCI
        nn_idx, _ = self.dci.query(X_query, num_neighbours=self.n_neighbors, 
            field_of_view=self.query_field_of_view, prop_to_retrieve=self.query_prop_to_retrieve, blind=True)

        nn_idx = np.array(nn_idx)
        X, y = self.X_db.view()[nn_idx], self.y_db.view()[nn_idx]

        y_pred = tf.get_default_session().run(self.y_pred, 
            feed_dict={self.X_linreg: X, self.y_linreg: y, self.X_query_linreg: X_query})

        return y_pred

    # updates the database and reconstructs DCI
    def fit(self, paths, targvals):
        X = np.concatenate([self._preproc(p) for p in paths])
        y = np.concatenate(targvals)

        if self.is_fit():
            logger.record_tabular("EVBefore", common.explained_variance(self._predict_linreg(X), y))

        self.X_db.add(X)
        self.y_db.add(y)

        self.dci = DCI(self.dim, self.num_comp_indices, self.num_simp_indices)
        self.dci.add(self.X_db.view(), num_levels=self.num_levels, 
            field_of_view=self.construction_field_of_view, prop_to_retrieve=self.construction_prop_to_retrieve)

        if self.is_fit():
            logger.record_tabular("EVAfter", common.explained_variance(self._predict_linreg(X), y))

    def is_fit(self):
        return self.X_db.size() >= self.n_neighbors

    def save(self, path, global_step):
        joblib.dump(self.X_db.view(), path + '_X-{}.pkl'.format(global_step))
        joblib.dump(self.y_db.view(), path + '_y-{}.pkl'.format(global_step))