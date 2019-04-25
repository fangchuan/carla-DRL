"""
*********************************************************************************************************
*
*	模块名称 : dqn_test模块
*	文件名称 : dqn_test.py
*	版    本 : V1.0.0
*   作    者 ：fc
*   日    期 ：2019/03/03
*	说    明 :
*             test the agent based on dqn.
*
* \par Method List:
*    1.    q_function(inpt, num_actions, scope, reuse=False);
*    2.    save_variables(save_path, variables=None, sess=None);
*    3.    load_variables(load_path, variables=None, sess=None);
*
*
*   修订记录：
	2019-03-03：   1.0.0      build;
							  position_tasks设定为straight_poses_Town01
							  修改calculate_reward(), 按REWARD_ASSIGN_PARAMETERS中的超参数来计算reward,
							  将forward_speed按speed_limit归一化为[0,100]范围, forward_speed-->forward_speed_post_process
							  把超速情况考虑进calculate_reward()内, 但是没有考虑速度一直为0的情况

    2019-03-05:   1.0.0       使用logger文件，plot csv文件观察训练过程
                              想使用MPI, 发现baselines中除了dqn和trpo_mpi，其他算法训练过程均使用了mpi来vectorize environnment.

    2019-03-06:   1.0.0       修改q_function网络的fc1层为512
   
    2019-04-24：  1.0.0       REPLAY_BUFFER_SIZE=1E5, 2E5会导致内存不够;
                              FINAL_EPSILON = 0.01, EVALUTE_EPSILON = 0.001; EXPLORATION_STEPS = 1E5;
                              UPDATE_TARGET_FREQUENCY = 3e3;


*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""
import gym
import numpy as np
import os
import sys
import baselines.common.tf_util as Util

from mpi4py import MPI
from datetime import datetime
from environment import carla_gym
from utils.common import DEBUG_PRINT as DEBUG_PRINT
from utils.common import common_arg_parser
from utils.common import NormalizedEnv, MaxAndSkipEnv
from utils import logger
from drl_algorithm.dqn import dqn
from utils.email_sender import email_sender

@email_sender
def main(arg):
    '''
    dqn_test.py
    '''
    argparser = common_arg_parser()
    args = argparser.parse_args()
    model_file_save_path = args.save_path
    model_file_load_path = args.load_path
    total_step_numbers = int(args.total_steps_num)
    is_play = args.play


    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        mpi_rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        mpi_rank = MPI.COMM_WORLD.Get_rank()

    model_file_name = datetime.now().strftime("carla-dqn-model-%Y-%m-%d-%H")
    model_file_name += '.ckpt'
    if not os.path.exists(model_file_save_path):
        os.makedirs(model_file_save_path, exist_ok=True)
    model_file_name = os.path.join(model_file_save_path, model_file_name)

    LEARN_RATE = 5e-4
    GAMMA = 0.99
    EXPERIENCE_REPLAY_BUFFER_SIZE = 100000
    EPSILON_EXPLORATION_TIMESTEPS = 100000
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.01
    EVALUATE_EPSILON = 0.001
    SAMPLE_BATCH_SIZE = 64
    STEPS_START_TRAIN = 1000
    UPDATE_TARGET_FREQUENCY = 1000
    CHECK_POINT_FREQUENCY = 20
    DQN_SCOPE = "deepq"


    with Util.make_session(num_cpu=8):

        # Create the environment
        env = gym.make("Carla-v0")
        assert env.config['discrete_actions'], print('DQN must be used in the discrete action space')
        # env = NormalizedEnv(env)
        # env = MaxAndSkipEnv(env)

        actor_fn = dqn.dqn(env= env,
                        total_step_numbers=total_step_numbers,
                        gamma=GAMMA,
                        lr=LEARN_RATE,
                        replay_buffer_size=EXPERIENCE_REPLAY_BUFFER_SIZE,
                        batch_size=SAMPLE_BATCH_SIZE,
                        exploration_steps=EPSILON_EXPLORATION_TIMESTEPS,
                        init_epsilon=INITIAL_EPSILON,
                        final_epsilon=FINAL_EPSILON,
                        start_steps=STEPS_START_TRAIN,
                        update_target_freq=UPDATE_TARGET_FREQUENCY,
                        checkpoints_freq=CHECK_POINT_FREQUENCY,
                        scope=DQN_SCOPE,
                        model_file_load_path=model_file_load_path,
                        model_file_save_path=model_file_name)
        if is_play:
            logger.log("Running trained model")
            from experiment_suit.carla_benchmark import benchmark_summary
            from functools import partial
            action_fn = partial(actor_fn, update_eps=EVALUATE_EPSILON)
            TEST_EPISODES = 10
            metrics = benchmark_summary(env, action_fn=action_fn, num_test_episodes=TEST_EPISODES, logger=logger)
            import json
            metrics_json = json.dumps(metrics, indent=4)
            with open('test_dqn_benchmark.json', 'w') as f:
                f.write(metrics_json)

        return actor_fn


if __name__ == '__main__':
    main(sys.argv)

