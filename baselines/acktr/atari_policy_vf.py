import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines import logger
import baselines.common as common
from sklearn.externals import joblib
from dci import DCI
import os
from baselines.acktr.numpy_deque import NumpyDeque

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class CnnLinregPolicyVF(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, 
                    timestep_window, n_neighbors): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (None, nh, nw, nc)
        nact = ac_space.n

        self.n_neighbors = n_neighbors
        self.X_db = NumpyDeque(max_capacity=timestep_window)
        self.y_db = NumpyDeque(max_capacity=timestep_window, one_dimensional=True)

        # prioritized DCI
        self.num_comp_indices = 2
        self.num_simp_indices = 7
        self.num_levels = 2
        self.construction_field_of_view = 10
        self.construction_prop_to_retrieve = 0.002
        self.query_field_of_view = 100
        self.query_prop_to_retrieve = 0.05
        self.dim = 512
        self.dci = DCI(self.dim, self.num_comp_indices, self.num_simp_indices)

        # graph for model
        X = tf.placeholder(tf.uint8, ob_shape) #obs

        with tf.variable_scope("model"):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()

        def get_nearest_neighbors(h):
            nn_idx, _ = self.dci.query(h, num_neighbours=self.n_neighbors, field_of_view=self.query_field_of_view,
                                        prop_to_retrieve=self.query_prop_to_retrieve, blind=True)

            nn_idx = np.array(nn_idx)
            X, y = self.X_db.view()[nn_idx], self.y_db.view()[nn_idx]

            return X, y

        # define the graph for linear regression
        X_linreg, y_linreg = tf.py_func(get_nearest_neighbors, [h], [tf.float32, tf.float32])
        ridge = tf.eye(self.dim) * 1e-2

        Xt_linreg = tf.transpose(X_linreg, [0, 2, 1])
        inv = tf.matrix_inverse(tf.matmul(Xt_linreg, X_linreg) + ridge)
        end = tf.matmul(Xt_linreg, tf.expand_dims(y_linreg, -1))
        weights = tf.squeeze(tf.matmul(inv, end), -1)
        vf = tf.reduce_sum(h * weights, -1)

        def step(ob, *_args, **_kwargs):
            if self.X_db.size() < self.n_neighbors:
                return sess.run(a0, {X:ob}), np.random.rand(ob.shape[0])
            return sess.run([a0, vf], {X:ob})

        def value(ob, *_args, **_kwargs):
            if self.X_db.size() < self.n_neighbors:
                return np.random.rand(ob.shape[0])
            return sess.run(vf, {X:ob})

        def fit_vf(X, y):
            self.X_db.add(X)
            self.y_db.add(y)

            self.dci = DCI(self.dim, self.num_comp_indices, self.num_simp_indices)
            self.dci.add(self.X_db.view(), num_levels=self.num_levels, 
                field_of_view=self.construction_field_of_view, prop_to_retrieve=self.construction_prop_to_retrieve)

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.fit_vf = fit_vf

        saver = tf.train.Saver(max_to_keep=150)

        def save(self, path, global_step):
            if not os.path.exists(path):
                os.makedirs(path)
            joblib.dump(self.X_db.view(), path + 'vf_X-{}.pkl'.format(global_step))
            joblib.dump(self.y_db.view(), path + 'vf_y-{}.pkl'.format(global_step))
            saver.save(tf.get_default_session(), path + 'pi', global_step=timesteps_so_far)

        self.save = save
