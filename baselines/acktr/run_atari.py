#!/usr/bin/env python3

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.acktr.atari_policy import CnnPolicy
from baselines.acktr.atari_policy_vf import CnnLinregPolicyVF


def train(env_id, num_timesteps, seed, num_cpu, run_number, timestep_window, n_neighbors):
    env = VecFrameStack(make_atari_env(env_id, num_cpu, seed), 4)
    policy_fn = CnnLinregPolicyVF
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu, 
            run_number=run_number, timestep_window=timestep_window, n_neighbors=n_neighbors)
    env.close()

def main():
    args = atari_arg_parser().parse_args()
    logger.configure()

    for run_number in range(1, 6):
    	train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_cpu=32, 
                run_number=run_number, timestep_window=100000, n_neighbors=200)
    	tf.reset_default_graph()

if __name__ == '__main__':
    main()
