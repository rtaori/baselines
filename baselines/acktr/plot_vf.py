## VALUE FUNCTION MONITORING
# step 3 - creating the plots

import numpy as np 
import matplotlib.pyplot as plt 

vf_of_base_obs = np.load('value_functions.npy')
vf_of_base_obs = np.rollaxis(vf_of_base_obs, axis=0, start=3)

print(vf_of_base_obs.shape)

for i in range(len(vf_of_base_obs)):
	for j in range(len(vf_of_base_obs[i])):
		plt.plot(vf_of_base_obs[i][j])
		plt.title('Value of State {} in Path {} Over Time'.format(j, i))
		plt.xlabel('Number of Timesteps')
		plt.ylabel('Expected Future Reward')
		plt.savefig('plots/state{}_path{}.png'.format(j, i))
		plt.close()