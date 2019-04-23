"""
*********************************************************************************************************
*
*	模块名称 : dqn模块
*	文件名称 : dqn.py
*	版    本 : V1.0.0
*   作    者 ：fc
*   日    期 ：2019/03/28
*	说    明 :
*             The functions in this file can are used to create the following functions:
*
* \par Method List:
*    1.    build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None);
*    2.    build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None, param_noise_filter_func=None);
*    3.    build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,double_q=True, scope="deepq", reuse=None, param_noise=False, param_noise_filter_func=None);
*    4.    scope_vars(scope, trainable_only=False);
*    5.    scope_name();
*    6.    absolute_scope_name(relative_scope_name);
*    7.    default_param_noise_filter(var);
*
*   修订记录：
	2019-03-28：   1.0.0      build;
							  修改q_function, 只有最后fc层后接BN;
    2019-3-30:                输入层加一层 BN;
                              最后fc层神经元个数改为64;

*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as Util

from collections import deque
from utils.replay_buffer import ReplayBuffer
from utils import logger
from utils.common import load_variables, save_variables
from utils.common import get_vars, count_vars
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from drl_algorithm.dqn import build_graph
from utils.common import batch_norm


def q_function(inpt, num_actions, scope, reuse=False):
    """
      This model takes as input an observation and returns Q_values of all actions
     :param inpt: [batch, height, width, channel] (NHWC)
     :param num_actions: the number of actions in the action space
     :param scope: name scope or variable scop
     :param reuse : True or False or tf.AUTO_REUSE
     :return: the probability of each action
     """
    batch_train = tf.constant(True, dtype=tf.bool)
    with tf.variable_scope(scope, reuse=reuse):
        out = tf.cast(inpt, tf.float32) / 255.
        out = layers.conv2d(out, 32, kernel_size=[8,8], stride=4, padding='SAME', activation_fn=None)
        out = batch_norm(out, train=batch_train, name="conv0_bn")
        out = tf.nn.relu(out)
        # out = layers.max_pool2d(out, kernel_size=[3,3], stride=2, padding='VALID')
        out = layers.conv2d(out, 64, kernel_size=[4,4], stride=2, padding='SAME', activation_fn=None)
        out = batch_norm(out, train=batch_train, name="conv1_bn")
        out = tf.nn.relu(out)
        # out = layers.max_pool2d(out, kernel_size=[3, 3], stride=2, padding='VALID')
        out = layers.conv2d(out, 64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
        out = batch_norm(out, train=batch_train, name="con2_bn")
        out = tf.nn.relu(out)
        reshape = tf.reshape(out, shape=[-1, out.get_shape()[1] * out.get_shape()[2] * 64])

        out = layers.fully_connected(reshape, num_outputs=512, activation_fn=None)
        out = batch_norm(out, train=batch_train, name="fc0_bn")
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, keep_prob=0.25)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def dqn(env,
        total_step_numbers=1000000,
        gamma=0.99,
        lr = 5e-4,
        replay_buffer_size=10000,
        prioritized_replay=False,
        batch_size=64,
        exploration_steps=10000,
        init_epsilon=1.0,
        final_epsilon=0.02,
        start_steps=1000,
        update_target_freq=1000,
        checkpoints_freq=20,
        scope='deepq',
        model_file_load_path=None,
        model_file_save_path=None,
        use_prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
         ):


    episodes = 0           #记录episode number
    mean_100ep_reward = 0  # 最后100episode的平均reward
    saved_mean_reward = None  # 保存的平均reward
    episode_reward = 0.0      #记录每个episode 的累计回报
    episode_rewards = deque(maxlen=100)


    # Create all the functions necessary to train the model
    actor, train, update_target, debug = build_graph.build_train(
        make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
        q_func=q_function,
        gamma=gamma,
        # grad_norm_clipping=10,
        # param_noise=True,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope=scope
    )

    # Create the replay buffer
    if not use_prioritized_replay:
        replay_buffer = ReplayBuffer(replay_buffer_size)
    else:
        replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_step_numbers
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                    initial_p=prioritized_replay_beta0,
                                    final_p=1.0)
    # Create the schedule for exploration starting from 1 (every action is random) down to
    # 0.02 (98% of actions are selected according to values predicted by the model).
    exploration = LinearSchedule(schedule_timesteps=exploration_steps,
                                 initial_p=init_epsilon,
                                 final_p=final_epsilon)

    # 全局参数初始化 并更新 target network.
    Util.initialize()
    update_target()

    # we need to reset the environment twice because of the bug of carla
    obs = env.reset()
    obs = env.reset_env()

    # 估计网络中变量数目：只占用显存的变量
    logger.info(" the network use {} varibales".format(count_vars(scope=scope)))

    # 保存和加载模型文件
    if model_file_load_path is not None:
        if os.path.exists(model_file_load_path):
            load_variables(model_file_load_path)
            logger.log('Loaded model from {}'.format(model_file_load_path))
        else:
            logger.error('Cannot find model file!!!')

    for step in range(total_step_numbers):
        # Take action and update exploration to the newest value
        action = actor(obs[None], update_eps=exploration.value(step))[0]
        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        episode_reward += rew
        if done:
            obs = env.reset_env()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            episodes += 1


        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if step > start_steps:
            if not use_prioritized_replay:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            else:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(step))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            # Update target network periodically.
            if step % update_target_freq == 0:
                update_target()

        if done and episodes % checkpoints_freq == 0:

            mean_100ep_reward = round(np.mean(episode_rewards), ndigits=2)

            logger.record_tabular("steps", step)
            logger.record_tabular("episodes", episodes)
            logger.record_tabular("mean_100ep_reward", mean_100ep_reward)
            logger.record_tabular("time_exploring", int(100 * exploration.value(step)))
            logger.dump_tabular()

            if (saved_mean_reward is None or mean_100ep_reward > saved_mean_reward):
                logger.log("Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                save_variables(model_file_save_path)
                saved_mean_reward = mean_100ep_reward

            obs = env.reset()


    return actor
