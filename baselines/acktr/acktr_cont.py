import numpy as np
import tensorflow as tf
from baselines import logger
import baselines.common as common
from baselines.common import tf_util as U
from baselines.acktr import kfac
from baselines.acktr.filters import ZFilter
from collections import deque
import matplotlib.pyplot as plt

def pathlength(path):
    return path["reward"].shape[0]# Loss function that we'll differentiate to get the policy gradient

def rollout(env, policy, max_pathlength, animate=False, obfilter=None):
    """
    Simulate the env and policy for max_pathlength steps
    """
    ob = env.reset()
    prev_ob = np.float32(np.zeros(ob.shape))
    # prev_ob = ob
    if obfilter: ob = obfilter(ob)
    terminated = False

    obs = []
    acs = []
    ac_dists = []
    logps = []
    rewards = []
    for _ in range(max_pathlength):
        if animate:
            env.render()
        state = np.concatenate([ob, prev_ob], -1)
        obs.append(state)
        ac, ac_dist, logp = policy.act(state)
        acs.append(ac)
        ac_dists.append(ac_dist)
        logps.append(logp)
        prev_ob = np.copy(ob)
        scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
        ob, rew, done, _ = env.step(scaled_ac)
        if obfilter: ob = obfilter(ob)
        rewards.append(rew)
        if done:
            terminated = True
            break
    return {"observation" : np.array(obs), "terminated" : terminated,
            "reward" : np.array(rewards), "action" : np.array(acs),
            "action_dist": np.array(ac_dists), "logp" : np.array(logps)}

def learn(env, policy, vf, gamma, lam, timesteps_per_batch, num_timesteps,
    animate=False, callback=None, desired_kl=0.002):
    obfilter = ZFilter(env.observation_space.shape)

    max_pathlength = env.spec.timestep_limit
    stepsize = tf.Variable(initial_value=np.float32(np.array(0.03)), name='stepsize')
    inputs, loss, loss_sampled = policy.update_info
    optim = kfac.KfacOptimizer(learning_rate=stepsize, cold_lr=stepsize*(1-0.9), momentum=0.9, kfac_update=2,\
                                epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                                weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    pi_var_list = []
    for var in tf.trainable_variables():
        if "pi" in var.name:
            pi_var_list.append(var)

    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_var_list)
    do_update = U.function(inputs, update_op)
    U.initialize()

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    # for qr in [q_runner, vf.q_runner]:
    for qr in [q_runner]:
        assert (qr != None)
        enqueue_threads.extend(qr.create_threads(tf.get_default_session(), coord=coord, start=True))

    # SAVING MODELS
    saver = tf.train.Saver(max_to_keep=150)
    # evaluating models
    # value_data = np.zeros((100, 4, len(range(2500, 662500+1, 7500))))
    # for i, model_end in enumerate(range(2500, 662500+1, 7500)):
    #     saver.restore(tf.get_default_session(), 'models/reacher_linear_kde/graph-{}'.format(model_end))
    #     vf.load_model('models/reacher_linear_kde/linear_kde-{}.pkl'.format(model_end))
    #     for init_state in range(100):
    #         print(model_end, init_state)
    #         env.env.curr_state = init_state
    #         avg_rewards_obs1, avg_rewards_obs2, est_value_obs1, est_value_obs2 = 0, 0, 0, 0
    #         for iter in range(5):
    #             path = rollout(env, policy, max_pathlength, animate=False, obfilter=obfilter)
    #             true_rewards = common.discount(path['reward'], gamma)
    #             pred_rewards = vf.predict(path)
    #             est_value_obs1 += pred_rewards[0]
    #             est_value_obs2 += pred_rewards[1]
    #             avg_rewards_obs1 += true_rewards[0]
    #             avg_rewards_obs2 += true_rewards[1]
    #         value_data[init_state, 0, i] = avg_rewards_obs1 / 5
    #         value_data[init_state, 1, i] = avg_rewards_obs2 / 5
    #         value_data[init_state, 2, i] = est_value_obs1 / 5
    #         value_data[init_state, 3, i] = est_value_obs2 / 5
    #     env.env.curr_state = None
    # np.save('value_estimates.npy', value_data)
    avg_vals, avg_vals_discounted, est_vals_linreg = [], [], []
    timesteps = []

    i = 0
    timesteps_so_far = 0
    while True:
        if timesteps_so_far >= num_timesteps:
            break
        logger.log("********** Iteration %i ************"%i)
        logger.log('Timesteps so far: ', timesteps_so_far)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            path = rollout(env, policy, max_pathlength, animate=(len(paths)==0 and (i % 10 == 0) and animate), obfilter=obfilter)
            paths.append(path)
            n = pathlength(path)
            timesteps_this_batch += n
            timesteps_so_far += n
            if timesteps_this_batch >= timesteps_per_batch:
                break

        # Estimate advantage function
        vtargs = []
        advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            vtargs.append(return_t)
            vpred_t = vf.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = rew_t + gamma*vpred_t[1:] - vpred_t[:-1]
            adv_t = common.discount(delta_t, gamma * lam)
            advs.append(adv_t)
        # Update value function
        vf.fit(paths, vtargs)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        oldac_dist = np.concatenate([path["action_dist"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        # Policy update
        do_update(ob_no, action_na, standardized_adv_n)

        ## SAVING MODELS
        if i % 3 == 0:
            saver.save(tf.get_default_session(), 'models/hopper_master/graph', global_step=timesteps_so_far)
            # vf.save_model('models/hopper_master/linreg-{}.pkl'.format(timesteps_so_far))
            avg_val, avg_val_discounted, est_val_linreg = 0, 0, 0
            for _ in range(5):
                path = rollout(env, policy, max_pathlength, animate=False, obfilter=obfilter)
                avg_val += path['reward'][1:].sum()
                avg_val_discounted += common.discount(path['reward'], gamma)[1]
                est_val_linreg += vf.predict(path)[1]
            avg_vals.append(avg_val / 5)
            avg_vals_discounted.append(avg_val_discounted / 5)
            est_vals_linreg.append(est_val_linreg / 5)
            timesteps.append(timesteps_so_far)
            plt.plot(timesteps, avg_vals, label='avg rewards', c='red')
            plt.plot(timesteps, avg_vals_discounted, label='avg discounted rewards', c='blue')
            plt.plot(timesteps, est_vals_linreg, label='linreg - est rewards', c='blue', linestyle='dashed')
            plt.title('Reward estimation for Hopper-v2')
            plt.xlabel('Number of timesteps')
            plt.ylabel('Reward')
            plt.legend()
            plt.savefig('plots/hopper_master/Hopper-v2.png')
            plt.close()

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)
        # Adjust stepsize
        kl = policy.compute_kl(ob_no, oldac_dist)
        if kl > desired_kl * 2:
            logger.log("kl too high")
            tf.assign(stepsize, tf.maximum(min_stepsize, stepsize / 1.5)).eval()
        elif kl < desired_kl / 2:
            logger.log("kl too low")
            tf.assign(stepsize, tf.minimum(max_stepsize, stepsize * 1.5)).eval()
        else:
            logger.log("kl just right!")

        logger.record_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("EpRewSEM", np.std([path["reward"].sum()/np.sqrt(len(paths)) for path in paths]))
        logger.record_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logger.record_tabular("KL", kl)
        if callback:
            callback()
        logger.dump_tabular()
        i += 1

    coord.request_stop()
    coord.join(enqueue_threads)