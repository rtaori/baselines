import os.path as osp
import os
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger
from baselines.common import set_global_seeds, explained_variance
from baselines.acktr.atari_runner import Runner
from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse
from baselines.acktr import kfac
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model(object):

    def __init__(self, policy_and_vf, ob_space, ac_space, nenvs,total_timesteps, num_processes=2, 
                envs_per_process=2, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', timestep_window=None, n_neighbors=None):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=num_processes)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        vf_coef = 0.0

        self.model2 = train_model = policy_and_vf(sess, ob_space, ac_space, 
                                    nenvs*nsteps, nsteps, timestep_window, n_neighbors)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        self.logits = logits = train_model.pi

        ##training loss
        pg_loss = tf.reduce_mean(ADV*logpac)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        train_loss = pg_loss + vf_coef * vf_loss

        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(
                            tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params=params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss,params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip,\
                momentum=0.9, kfac_update=1, epsilon=0.01,\
                stats_decay=0.99, async=1, cold_iter=10, max_grad_norm=max_grad_norm)

            update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, PG_LR:cur_lr}

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )

            return policy_loss, value_loss, policy_entropy

        def save(save_path, global_step):
            train_model.save(save_path, global_step)

        self.train = train
        self.save = save
        self.train_model = train_model
        self.step = train_model.step
        self.value = train_model.value
        self.value_nn = train_model.value_nn
        self.value_linreg = train_model.value_linreg
        self.get_time_back = train_model.get_time_back

        tf.global_variables_initializer().run(session=sess)

def learn(policy_and_vf, envs, env_id, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, 
            num_processes=2, envs_per_process=2,
          nsteps=20, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5, kfac_clip=0.001, 
          save_interval=None, lrschedule='linear', run_number=None, timestep_window=None, n_neighbors=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = num_processes * envs_per_process
    ob_space = envs[0].observation_space
    ac_space = envs[0].action_space
    make_model = lambda : Model(policy_and_vf, ob_space, ac_space, nenvs, total_timesteps, 
                                num_processes=num_processes, envs_per_process=envs_per_process,
                                nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef, 
                                lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip, lrschedule=lrschedule, 
                                timestep_window=timestep_window, n_neighbors=n_neighbors)

    logger.configure(dir='baselines/testing/{}/run{}/log.csv'.format(env_id, run_number))
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    runners = [Runner(env, model, nsteps=nsteps, gamma=gamma) for env in envs]
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)

    # SAVING MODELS
    avg_vals, avg_vals_discounted, est_vals_orig, est_vals_nn, est_vals_linreg = [], [], [], [], []
    timesteps = []
    time_backs = []

    for update in range(total_timesteps//nbatch+1):

        obs, rewards, masks, actions, values, values_nn, values_linreg, summed_rewards, mb_rewards, last_values, last_obs = runners[0].run()
        for i in range(1, len(runners)):
            obs_, rewards_, masks_, actions_, values_, values_nn_, values_linreg_, summed_rewards_, mb_rewards_, last_values_, last_obs_ = runners[i].run()
            obs = np.concatenate([obs, obs_])
            rewards = np.concatenate([rewards, rewards_])
            masks = np.concatenate([masks, masks_])
            actions = np.concatenate([actions, actions_])
            values = np.concatenate([values, values_])
            values_nn = np.concatenate([values_nn, values_nn_])
            values_linreg = np.concatenate([values_linreg, values_linreg_])
            summed_rewards.extend(summed_rewards_)
            mb_rewards = np.concatenate([mb_rewards, mb_rewards_])
            last_values = np.concatenate([last_values, last_values_])
            last_obs = np.concatenate([last_obs, last_obs_])

        if model.train_model.is_vf_fit():
            time_backs.append(model.get_time_back(obs, update))

        flag = False
        if model.train_model.is_vf_fit():
            flag = True
            policy_loss, value_loss, policy_entropy = model.train(obs, 
                rewards.flatten(), masks.flatten(), actions.flatten(), values.flatten())
        model.train_model.fit_vf(obs, rewards.flatten(), update)

        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values.flatten(), rewards.flatten())
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            if flag:
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
        
        ## SAVING MODELS
        save_path = 'testing/{}/run{}/'.format(env_id, run_number)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # if update % 50 == 0:
        #     model.save(save_path, update*nbatch)
        #     final_activations = model.get_last_activations(obs)
        #     joblib.dump(final_activations, save_path+'h-{}.pkl'.format(update*nbatch))

        avg_val = np.mean(summed_rewards)
        avg_val_discounted = rewards[:, 0].mean()
        est_val_orig = values[:, 0].mean()
        est_val_nn = values_nn[:, 0].mean()
        est_val_linreg = values_linreg[:, 0].mean()
        avg_vals.append(avg_val)
        avg_vals_discounted.append(avg_val_discounted)
        est_vals_orig.append(est_val_orig)
        est_vals_nn.append(est_val_nn)
        est_vals_linreg.append(est_val_linreg)
        timesteps.append(update*nbatch)

        # joblib.dump(obs, save_path+'obs-{}.pkl'.format(update*nbatch))
        # joblib.dump(rewards, save_path+'rewards-{}.pkl'.format(update*nbatch))
        # joblib.dump(actions, save_path+'actions-{}.pkl'.format(update*nbatch))
        # joblib.dump(values, save_path+'values-{}.pkl'.format(update*nbatch))
        # joblib.dump(summed_rewards, save_path+'summed_rewards-{}.pkl'.format(update*nbatch))
        # joblib.dump(mb_rewards, save_path+'mb_rewards-{}.pkl'.format(update*nbatch))
        # joblib.dump(last_values, save_path+'last_values-{}.pkl'.format(update*nbatch))
        # joblib.dump(last_obs, save_path+'last_obs-{}.pkl'.format(update*nbatch))

        joblib.dump(time_backs, save_path+'time_backs.pkl')
        joblib.dump(avg_vals, save_path+'avg_vals.pkl')
        joblib.dump(avg_vals_discounted, save_path+'avg_vals_discounted.pkl')
        joblib.dump(est_vals_orig, save_path+'est_vals_orig.pkl')
        joblib.dump(est_vals_nn, save_path+'est_vals_nn.pkl')
        joblib.dump(est_vals_linreg, save_path+'est_vals_linreg.pkl')
        joblib.dump(timesteps, save_path+'timesteps.pkl')

        # plt.plot(timesteps, avg_vals, label='avg rewards', c='red')
        plt.plot(timesteps, avg_vals_discounted, label='avg discounted rewards', c='red')
        plt.plot(timesteps, est_vals_orig, label='orig - est rewards', c='blue')
        plt.plot(timesteps, est_vals_nn, label='nn - est rewards', c='green')
        plt.plot(timesteps, est_vals_linreg, label='linreg - est rewards', c='yellow')
        plt.title('Reward estimation for {}'.format(env_id))
        plt.xlabel('Number of timesteps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(save_path + 'plot.png')
        plt.close()

    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()
