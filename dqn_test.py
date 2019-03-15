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


*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
import sys
from mpi4py import MPI
from collections import deque

import baselines.common.tf_util as Util
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
from datetime import datetime
from environment import carla_gym
from drl_algorithm.dqn import build_graph
from utils.common import DEBUG_PRINT as DEBUG_PRINT
from utils.common import common_arg_parser
from utils.common import NormalizedEnv
from utils.common import load_variables, save_variables
from utils.common import get_vars, count_vars
from utils import logger




def q_function(inpt, num_actions, scope, reuse=False):
    """
     q_function模型
      This model takes as input an observation and returns values of all actions
     :param inpt: [batch, height, width, channel] (NHWC)
     :param num_actions: the number of actions in the action space
     :param scope: name scope or variable scop
     :param reuse : True or False or tf.AUTO_REUSE
     :return: the probability of each action
     """

    with tf.variable_scope(scope, reuse=reuse):
        out = tf.cast(inpt, tf.float32) / 255.
        # batch_norm_params = {
        #     'decay': 0.997,
        #     'epsilon': 1e-5,
        #     'scale': True,
        #     'is_training':True,
        #     'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # }
        out = layers.conv2d(out, 32, kernel_size=[8,8], stride=4, padding='SAME', activation_fn=tf.nn.relu)
                            # normalizer_fn=layers.batch_norm, normalizer_params=batch_norm_params)
        out = layers.max_pool2d(out, kernel_size=[3,3], stride=2, padding='VALID')
        out = layers.conv2d(out, 64, kernel_size=[4,4], stride=2, padding='SAME', activation_fn=tf.nn.relu)
                            # normalizer_fn=layers.batch_norm, normalizer_params=batch_norm_params)
        out = layers.max_pool2d(out, kernel_size=[3, 3], stride=2, padding='VALID')
        out = layers.conv2d(out, 192, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
                            # normalizer_fn=layers.batch_norm, normalizer_params=batch_norm_params)
        reshape = tf.reshape(out, shape=[-1, out.get_shape()[1] * out.get_shape()[2] * 192])

        out = layers.fully_connected(reshape, num_outputs=512, activation_fn=tf.nn.relu)
        # out = tf.layers.batch_normalization(out, training=batch_norm_params['is_training'])
        # out = tf.nn.sigmoid(out)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out




def main(arg):

    argparser = common_arg_parser()
    args = argparser.parse_args()
    model_file_save_path = args.save_path
    model_file_load_path = args.load_path
    total_step_numbers = int(args.total_steps_num)
    is_play = args.play

    LEARN_RATE = 5e-4
    GAMMA = 0.99
    EXPERIENCE_REPLAY_BUFFER_SIZE = 50000
    EPSILON_EXPLORATION_TIMESTEPS = 10000
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.02
    SAMPLE_BATCH_SIZE = 32
    STEPS_START_TRAIN = 1000
    UPDATE_TARGET_FREQUENCY = 1000
    MAX_ACCUMULATED_REWARDS = 20.0
    CHECK_POINT_FREQUENCY = 20
    DQN_SCOPE = "deepq"

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        mpi_rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        mpi_rank = MPI.COMM_WORLD.Get_rank()

    with Util.make_session(num_cpu=8):

        episode_rewards = deque(maxlen=100)
        episode_reward = 0.0
        episodes = 1
        mean_100ep_reward = 0  # 最后100episode的平均reward
        saved_mean_reward = None  # 保存的平均reward
        model_file_name = datetime.now().strftime("carla-dqn-model-%Y-%m-%d-%H")
        model_file = model_file_name + '.ckpt'

        # Create the environment
        env = gym.make("Carla-v0")
        assert env.config['discrete_actions'], DEBUG_PRINT('DQN must be used in the discrete action space')
        # env = NormalizedEnv(env)

        # Create all the functions necessary to train the model
        actor, train, update_target, debug = build_graph.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=q_function,
            gamma=GAMMA,
            # grad_norm_clipping=10,
            # param_noise=True,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=LEARN_RATE),
            scope=DQN_SCOPE
        )

        # Create the replay buffer
        replay_buffer = ReplayBuffer(EXPERIENCE_REPLAY_BUFFER_SIZE)
        # prioritized_replay_alpha = 0.6
        # prioritized_replay_beta0 = 0.4
        # prioritized_replay_beta_iters = None
        # prioritized_replay_eps = 1e-6
        # replay_buffer = PrioritizedReplayBuffer(50000, alpha=prioritized_replay_alpha)
        # if prioritized_replay_beta_iters is None:
        #     prioritized_replay_beta_iters = args.total_steps_num
        # beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
        #                                initial_p=prioritized_replay_beta0,
        #                                final_p=1.0)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=EPSILON_EXPLORATION_TIMESTEPS,
                                     initial_p=INITIAL_EPSILON,
                                     final_p=FINAL_EPSILON)

        # 全局参数初始化 并更新 target network.
        Util.initialize()
        update_target()
        obs = env.reset()

        # 估计网络中变量数目：只占用显存的变量
        logger.info(" the network use {} varibales".format(count_vars(DQN_SCOPE)))

        # 保存和加载模型文件
        if not os.path.exists(model_file_save_path):
            os.makedirs(model_file_save_path, exist_ok=True)
        model_file = os.path.join(model_file_save_path, model_file)
        if args.load_path is not None:
            load_variables(model_file_load_path)
            logger.log('Loaded model from {}'.format(model_file_load_path))
            logger.info('Using agent with the following configuration of train:')
            logger.info(str(train.__dict__.items()))

        for step in range(total_step_numbers):
            # Take action and update exploration to the newest value
            action = actor(obs[None], update_eps=exploration.value(step))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_reward += rew
            if done:
                obs = env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                episodes += 1

            mean_100ep_reward = round(np.mean(episode_rewards), ndigits=2)

            is_solved = mean_100ep_reward >= MAX_ACCUMULATED_REWARDS
            if is_solved:
                print(" steps = {}, mean_episode_reward = {}".format(step, mean_100ep_reward))
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if step > STEPS_START_TRAIN:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(SAMPLE_BATCH_SIZE)
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # experience = replay_buffer.sample(32, beta=beta_schedule.value(step))
                    # (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    #
                    # td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                    # new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    # replay_buffer.update_priorities(batch_idxes, new_priorities)
                # Update target network periodically.
                if step % UPDATE_TARGET_FREQUENCY == 0:
                    update_target()

            if done and episodes % CHECK_POINT_FREQUENCY == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", episodes)
                logger.record_tabular("mean_100ep_reward", mean_100ep_reward)
                logger.record_tabular("time_exploring", int(100 * exploration.value(step)))
                logger.dump_tabular()

                if (saved_mean_reward is None or mean_100ep_reward > saved_mean_reward) and (mpi_rank == 0):
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward,
                                                                                           mean_100ep_reward))
                    save_variables(model_file)
                    saved_mean_reward = mean_100ep_reward

        if is_play:
            logger.log("Running trained model")
            obs = env.reset()

            while True:
                action = actor(obs[None], update_eps=exploration.value(step))[0]
                obs, _, done, _ = env.step(action)
                done = done.any() if isinstance(done, np.ndarray) else done

                if done:
                    obs = env.reset()

        return actor


if __name__ == '__main__':
    main(sys.argv)

