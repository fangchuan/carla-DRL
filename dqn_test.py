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


*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""
import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tempfile
# from tensorboardX import SummaryWriter

import baselines.common.tf_util as Util
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
from environment import carla_gym
from utils.common import DEBUG_PRINT as DEBUG_PRINT

from argparse import ArgumentParser
import os
import joblib

MAX_STEPS = 1e6
MAX_ACCUMULATED_REWARDS = 20.0
CHECK_POINT_FREQUENCY = 20



# observation shape: (224, 224, 6)
# q_function/Conv/Relu [2, 56, 56, 32]
# q_function/MaxPool2D/MaxPool [2, 27, 27, 32]
# q_function/Conv_1/Relu [2, 14, 14, 64]
# q_function/MaxPool2D_1/MaxPool [2, 6, 6, 64]
# q_function/Conv_2/Relu [2, 6, 6, 192]
#
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
        out = inpt
        out = layers.conv2d(inpt, 32, kernel_size=[11,11], stride=4, padding='SAME', activation_fn=tf.nn.relu) #normalizer_fn=tf.nn.batch_normalization)
        out = layers.max_pool2d(out, kernel_size=[3,3], stride=2, padding='VALID')
        out = layers.conv2d(out, 64, kernel_size=[5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
        out = layers.max_pool2d(out, kernel_size=[3, 3], stride=2, padding='VALID')
        out = layers.conv2d(out, 192, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
        reshape = tf.reshape(out, shape=[-1, out.get_shape()[1] * out.get_shape()[2] * 192])
        out = layers.fully_connected(reshape, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def save_variables(save_path, variables=None, sess=None):
    """
     保存模型参数

     :param save_path: the path to the model file
     :param variables: the trainable variables in the graph
     :param sess: the session of the graph
     :return: None
     """
    sess = sess or tf.get_default_session()
    variables = variables or tf.trainable_variables()

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)

def load_variables(load_path, variables=None, sess=None):
    """
     加载模型参数

     :param load_path: the path to the model file
     :param variables: the trainable variables in the graph
     :param sess: the session of the graph
     :return: None
     """
    sess = sess or tf.get_default_session()
    variables = variables or tf.trainable_variables()

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)



if __name__ == '__main__':

    argparser = ArgumentParser("dqn_agent")
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--params-file',
        metavar='params-file',
        default='dqn_parameters.json',
        help='path to parameters file.Default=qdn_parameters.json')
    argparser.add_argument(
        '--save-path',
        default="trained_models/",
        metavar="save_path",
        help="directory to save/load trained model. Default= ./trained_models/")
    argparser.add_argument(
        "--load-path",
        default=None,
        metavar='load_path',
        help="directory to load trained model. Default= ./trained_models/carla-dqn-model.ckpt")
    argparser.add_argument(
        '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-g', '--gpu-id',
        type=int,
        default=0,
        metavar="GPU_ID",
        help='GPU device ID to use. Default:0')

    args = argparser.parse_args()

    with Util.make_session(num_cpu=8):

        episode_rewards = [0.0]
        mean_100ep_reward = 0   #最后100episode的平均reward
        saved_mean_reward = None   #保存的平均reward
        check_point_freq = CHECK_POINT_FREQUENCY
        model_file = "carla-dqn-model.ckpt"

        # Create the environment
        env = gym.make("Carla-v0")

        assert env.config['discrete_actions'], DEBUG_PRINT('DQN must be used in the discrete action space')

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=q_function,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # 全局参数初始化 并更新 target network.
        Util.initialize()
        update_target()
        obs = env.reset()

        if os.path.exists(args.save_path):
            model_file = os.path.join(args.save_path, model_file)

        if args.load_path is not None :
            load_variables(args.load_path)
            logger.log('Loaded model from {}'.format(args.load_path))

        for step in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(step))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]),2)

            is_solved = step > MAX_STEPS or mean_100ep_reward >= MAX_ACCUMULATED_REWARDS
            if is_solved:
                print(" steps = {}, mean_episode_reward = {}".format(step, mean_100ep_reward))
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if step > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if step % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % check_point_freq == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(step)))
                logger.dump_tabular()

                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    saved_mean_reward = mean_100ep_reward

