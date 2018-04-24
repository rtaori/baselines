#!/usr/bin/env python3

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.acktr.atari_policy_vf import CnnLinregPolicyVF


def train(env_id, num_timesteps, seed, num_processes, envs_per_process, run_number, timestep_window, n_neighbors):
    env = VecFrameStack(make_atari_env(env_id, num_processes, seed), 4)
    policy_fn = CnnLinregPolicyVF
    learn(policy_fn, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), num_processes=num_processes, 
            envs_per_process=envs_per_process, run_number=run_number, timestep_window=timestep_window, 
            n_neighbors=n_neighbors)
    env.close()

def main():
    args = atari_arg_parser().parse_args()
    logger.configure()

    for run_number in range(1, 6):
    	train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_processes=8, envs_per_process=4, 
                run_number=run_number, timestep_window=100000, n_neighbors=200)
    	tf.reset_default_graph()

if __name__ == '__main__':
    main()
