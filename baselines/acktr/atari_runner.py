import numpy as np
from baselines.a2c.utils import discount_with_dones

class Runner(object):

    def __init__(self, env, model, nsteps, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.obs = np.random.randint(0, high=200, size=(nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_nn_values, mb_linreg_values = [], []
        for n in range(self.nsteps):
            actions, values = self.model.step(self.obs, self.dones)
            nn_values = self.model.value_nn(self.obs, self.dones)
            linreg_values = self.model.value_linreg(self.obs, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_nn_values.append(nn_values)
            mb_linreg_values.append(linreg_values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mb_rewards = mb_rewards.copy()
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_nn_values = np.asarray(mb_nn_values, dtype=np.float32).swapaxes(1, 0)
        mb_linreg_values = np.asarray(mb_linreg_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.dones).tolist()
        last_values_nn = self.model.value_nn(self.obs, self.dones).tolist()
        last_values_linreg = self.model.value_linreg(self.obs, self.dones).tolist()
        summed_rewards []
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            dones = dones.tolist()
            if dones[-1] == 0:
                summed_rewards.append(rewards.sum() + value)
                rewards = discount_with_dones(rewards.tolist()+[value], dones+[0], self.gamma)[:-1]
            else:
                summed_rewards.append(rewards.sum())
                rewards = discount_with_dones(rewards.tolist(), dones, self.gamma)
            mb_rewards[n] = rewards
        return mb_obs, mb_rewards, mb_masks, mb_actions, mb_values, mb_nn_values, mb_linreg_values, summed_rewards, mb_mb_rewards, last_values, self.obs
