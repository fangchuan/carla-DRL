"""
*********************************************************************************************************
*
*	模块名称 : ddpg_test模块
*	文件名称 : ddpg_test.py
*	版    本 : V1.0.0
*   作    者 ：fc
*   日    期 ：2019/03/11
*	说    明 :
*             test the agent based on ddpg.
*
* \par Method List:
*    1.    actor_critic_network(**net_kwargs);
*    2.    learn(network, env,
          seed=None,
          total_steps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          memory_size= 1e4,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          model_file_load_path=None,
          model_file_save_path=None,
          **network_kwargs);
*
*
*
*   修订记录：
	2019-03-11：   1.0.0      build;
	                         修改ddpg算法，使ddpg可以运行在Carla-v0上;
	                         param_noise好像还有问题;

    2019-03-13:   1.0.0      revise learn函数, 将eval_env相关代码删除;
                             添加argumentParser, 增加save_model, load_model, play功能(未测试);

    2019-03-15:   1.0.0      使用自定义network, 每一层后跟layer_norm, 最后的激活函数换为tanh;
                             使用ReplayBuffer替换Memory;

*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""
import sys
import os
import time
import gym
import numpy as np
import pickle
import tensorflow as tf
import baselines.common.tf_util as Util
import tensorflow.contrib.layers as layers
from datetime import datetime

from collections import deque
from drl_algorithm.ddpg.ddpg import DDPG
from drl_algorithm.ddpg.model import Actor, Critic
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
from baselines.common.models import register
from utils import logger
from environment import carla_gym
from utils.common import load_variables, save_variables
from utils.common import common_arg_parser
from utils.common import get_vars, count_vars
from utils.replay_buffer import ReplayBuffer
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


@register("my_actor_critic_net")
def actor_critic_network(**net_kwargs):
    def actor_critic_net(x):
        out = tf.cast(x, dtype=tf.float32) / 255.0
        out = layers.conv2d(out, 32, kernel_size=[8, 8], stride=4, padding='SAME', activation_fn=tf.nn.relu)  # normalizer_fn=tf.nn.batch_normalization)
        out = layers.max_pool2d(out, kernel_size=[3, 3], stride=2, padding='VALID')
        out = layers.layer_norm(out, center=True, scale=True)
        out = layers.conv2d(out, 64, kernel_size=[4, 4], stride=2, padding='SAME', activation_fn=tf.nn.relu)
        out = layers.max_pool2d(out, kernel_size=[3, 3], stride=2, padding='VALID')
        out = layers.layer_norm(out, center=True, scale=True)
        out = layers.conv2d(out, 64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
        out = layers.layer_norm(out, center=True, scale=True)
        reshape = tf.reshape(out, shape=[-1, out.get_shape()[1] * out.get_shape()[2] * 64])
        out = layers.fully_connected(reshape, num_outputs=512, activation_fn=None)
        out = layers.layer_norm(out, center=True, scale=True)
        out = tf.nn.tanh(out)
        return out
    return actor_critic_net


def learn(network, env,
          seed=None,
          total_steps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          memory_size= 1e4,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          model_file_load_path=None,
          model_file_save_path=None,
          **network_kwargs):

    set_global_seeds(seed)

    if total_steps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_steps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        mpi_rank = MPI.COMM_WORLD.Get_rank()
    else:
        mpi_rank = 0

    number_actions = env.action_space.shape[-1]
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    obs_range = (env.observation_space.low, env.observation_space.high)
    action_range = (env.action_space.low, env.action_space.high)

    # random replay buffer
    # memory = Memory(limit=int(memory_size), action_shape=action_shape, observation_shape=obs_shape)
    memory = ReplayBuffer(int(memory_size))
    # critic网络
    critic = Critic(name="critic", network=network,**network_kwargs)
    # actor网络
    actor = Actor(number_actions, name="actor", network=network, **network_kwargs)

    # 添加探索策略: 使用action_noise 或param_noise
    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(number_actions), sigma=float(stddev) * np.ones(number_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(number_actions), sigma=float(stddev) * np.ones(number_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    agent = DDPG(actor, critic, memory, obs_shape, action_shape,
                 gamma=gamma,
                 tau=tau,
                 normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 observation_range=obs_range,
                 action_range=action_range,
                 batch_size=batch_size,
                 action_noise=action_noise,
                 param_noise=param_noise,
                 critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr,
                 critic_lr=critic_lr,
                 enable_popart=popart,
                 clip_norm=clip_norm,
                 reward_scale=reward_scale)

    sess = Util.get_session()
    # 初始化全局变量, 初始化目标网络 ,如果采用分布式并行训练, 同步optimizer
    agent.initialize(sess)
    sess.graph.finalize()

    # 估计网络中变量数目：只占用显存的变量
    logger.info(" the network use {} varibales".format(count_vars("critic")))
    logger.info(" the network use {} varibales".format(count_vars("actor")))


    # 加载模型文件
    if model_file_load_path is not None:
        if os.path.exists(model_file_load_path):
            load_variables(model_file_load_path)
            logger.log('Loaded model from {}'.format(model_file_load_path))
        else:
            logger.error("Cannont find your model file!")

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # reset agent internal state
    agent.reset()
    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()

    steps = 0           #agent目前总共步数
    episode_reward = 0.0  #每一幕的累计回报
    episode_step = 0  #每一幕走的累计步数
    episodes = 0     #scalar, 总共的episode数量
    start_time = time.time()

    saved_mean_reward = None  # 保存的平均reward
    epoch_episode_rewards = deque(maxlen=100)  #记录每一幕的累计回报
    epoch_episode_steps = deque(maxlen=100)  #记录每一幕的累计步数
    epoch_actions = []  #记录所有的action
    epoch_qs = deque(maxlen=100)     #记录所有的q值
    epoch_actor_losses = deque(maxlen=100)  #记录actor网络的损失值
    epoch_critic_losses = deque(maxlen=100)  #记录critic网络的损失值
    epoch_adaptive_distances = deque(maxlen=100)  #记录adaptive peturbed actor与actor之间的MSE

    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            # if nenvs > 1:
            #     # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
            #     # of the environments, so resetting here instead
            #     agent.reset()
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q, _, _ = agent.step(np.array(obs)[None], apply_noise=True, compute_Q=True)
                print("action : ", action)
                action = action.reshape(number_actions)
                # Execute next action.
                # if mpi_rank == 0 and render:
                #     env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                # note these outputs are batched from vecenv

                steps += 1
                # if mpi_rank == 0 and render:
                #     env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                # 像replay buffer中添加观测数据
                agent.store_transition(np.array(obs)[None], np.array(action)[None],
                                       np.array(r)[None], np.array(new_obs)[None], np.array(done)[None]) #the batched data will be unrolled in memory.py's append.

                obs = new_obs

                if done :
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    episode_reward = 0.
                    episode_step = 0
                    episodes += 1
                    # reset environment和agent
                    obs = env.reset()
                    agent.reset()

            # Train.
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if len(memory) >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()

        mean_100ep_reward = np.mean(epoch_episode_rewards)
        combined_stats['rollout/mean_100ep_reward'] = mean_100ep_reward
        combined_stats['rollout/mean_100ep_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/mean_100ep_Q'] = np.mean(epoch_qs)
        combined_stats['train/100_loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/100_loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_100distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(steps) / float(duration)
        combined_stats['total/episodes'] = episodes
        # combined_stats['rollout/actions_std'] = np.std(epoch_actions)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = steps

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if mpi_rank == 0:
            logger.dump_tabular()

        if (saved_mean_reward is None or mean_100ep_reward > saved_mean_reward) and (mpi_rank == 0):
            logger.log("Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward,
                                                                                   mean_100ep_reward))
            save_variables(model_file_save_path)
            saved_mean_reward = mean_100ep_reward

    return agent


def main(argvs):
    argparser = common_arg_parser()
    args = argparser.parse_args()
    model_file_save_path = args.save_path
    model_file_load_path = args.load_path
    total_step_numbers = args.total_steps_num
    is_play = args.play

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    with Util.make_session(num_cpu=8):
        model_file_name = datetime.now().strftime("carla-ddpg-model-%Y-%m-%d-%H")
        model_file = model_file_name + '.ckpt'

        env = gym.make('Carla-v0')

        if not os.path.exists(model_file_save_path):
            os.makedirs(model_file_save_path, exist_ok=True)
        model_file = os.path.join(model_file_save_path, model_file)

        agent = learn(network='my_actor_critic_net',
                      env=env,
                      seed=12345,
                      total_steps=total_step_numbers,
                      noise_type="adaptive-param_0.2",
                      actor_lr=1e-3,
                      critic_lr=1e-3,
                      model_file_load_path=model_file_load_path,
                      model_file_save_path=model_file)

        if is_play:
            logger.log("Running DDPG trained model")
            obs = env.reset()
            steps = 0
            while True:
                action, _, _, _ = agent.step(np.array(obs)[None], apply_noise=False, compute_Q=True)
                print("action : ", action)
                number_actions = env.action_space.shape[-1]
                action = action.reshape(number_actions)

                max_action = env.action_space.high
                obs, reward, done, info = env.step(max_action * action)
                done = done.any() if isinstance(done, np.ndarray) else done
                steps += 1

                logger.record_tabular("steps", steps)
                logger.record_tabular("reward", reward)

                if done:
                    obs = env.reset()
                    logger.dump_tabular()

        return agent



if __name__ == '__main__':
    main(sys.argv)
