## VALUE FUNCTION MONITORING
# step 3 - creating the plots

import numpy as np 
import matplotlib.pyplot as plt 

# vf_of_base_obs = np.load('value_functions.npy')
# vf_of_base_obs = np.rollaxis(vf_of_base_obs, axis=0, start=3)

# print(vf_of_base_obs.shape)

# for i in range(len(vf_of_base_obs)):
# 	for j in range(len(vf_of_base_obs[i])):
# 		plt.plot(vf_of_base_obs[i][j])
# 		plt.title('Value of State {} in Path {} Over Time'.format(j, i))
# 		plt.xlabel('Number of Timesteps')
# 		plt.ylabel('Expected Future Reward')
# 		plt.savefig('plots/state{}_path{}.png'.format(j, i))
# 		plt.close()


value_data = np.load('plots/reacher-plots-linear-kde/value_estimates.npy')
for init_state in range(value_data.shape[0]):
	plt.plot(range(2500, 662500+1, 7500), value_data[init_state][0], label='avg rollout rewards for init state', c='blue')
	plt.plot(range(2500, 662500+1, 7500), value_data[init_state][2], label='estimated rewards for init state', c='blue', linestyle='dashed')
	plt.plot(range(2500, 662500+1, 7500), value_data[init_state][1], label='avg rollout rewards for state after init', c='red')
	plt.plot(range(2500, 662500+1, 7500), value_data[init_state][3], label='estimated rewards for state after init', c='red', linestyle='dashed')
	plt.title('Reward estimation for state {}'.format(init_state))
	plt.xlabel('Number of iterations')
	plt.ylabel('Reward')
	plt.legend()
	plt.savefig('plots/reacher-plots-linear-kde/state{}.png'.format(init_state))
	plt.close()