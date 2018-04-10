import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance

from baselines.a2c.a2c import Runner
from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse
from baselines.acktr import kfac
from collections import deque

import matplotlib.pyplot as plt


class Model(object):

    def __init__(self, policy, value_fn, ob_space, ac_space, nenvs,total_timesteps, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        self.model = step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        self.model2 = train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)
        self.value_fn = value_fn(n_neighbors=200, sess=self.sess)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        self.logits = logits = train_model.pi

        ##training loss
        pg_loss = tf.reduce_mean(ADV*logpac)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        pg_loss = pg_loss - ent_coef * entropy
        # vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        vf_loss = tf.placeholder(tf.float32, shape=[])
        train_loss = pg_loss + vf_coef * vf_loss


        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        # sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        # self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss #+ vf_fisher_loss

        self.params=params = find_trainable_variables("model")
        self.params = params = self.params[:-2]

        self.grads_check = grads = tf.gradients(train_loss,params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip,\
                momentum=0.9, kfac_update=1, epsilon=0.01,\
                stats_decay=0.99, async=1, cold_iter=10, max_grad_norm=max_grad_norm)

            update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        self.get_values = lambda obs: self.value_fn.predict(obs)
        self.value_loss = lambda obs, rewards: np.square(self.get_values(obs) - rewards).mean()

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            value_loss = self.value_loss(obs, rewards)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, PG_LR:cur_lr, vf_loss:value_loss}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, policy_entropy, _ = sess.run(
                [pg_loss, entropy, train_op],
                td_map
            )
            self.value_fn.fit(obs, rewards)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

def learn(policy, value, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda : Model(policy, value, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
                                =nsteps, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()


    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)

    avg_vals, avg_vals_discounted, est_vals_linreg = [], [], []
    timesteps = []
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, undiscounted_rewards = runner.run()
        print('finished running')
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards.flatten(), masks.flatten(), actions.flatten(), values.flatten())
        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values.flatten(), rewards.flatten())
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()


        avg_val, avg_val_discounted, est_val_linreg = undiscounted_rewards[:, 0].mean(), rewards[:, 0].mean(), values[:, 0].mean()
        avg_vals.append(avg_val), avg_vals_discounted.append(avg_val_discounted), est_vals_linreg.append(est_val_linreg)
        timesteps.append(update*nbatch)
        plt.plot(timesteps, avg_vals, label='avg rewards', c='red')
        plt.plot(timesteps, avg_vals_discounted, label='avg discounted rewards', c='blue')
        plt.plot(timesteps, est_vals_linreg, label='linreg - est rewards', c='blue', linestyle='dashed')
        plt.title('Reward estimation for BreakoutNoFrameskip-v2')
        plt.xlabel('Number of timesteps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('plots/BreakoutNoFrameskip_master/BreakoutNoFrameskip-v2.png')
        plt.close()


        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()
