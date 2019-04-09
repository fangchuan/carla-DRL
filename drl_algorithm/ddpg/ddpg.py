import os
import gym
import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from datetime import datetime
from collections import deque
from environment import carla_gym
from utils.common import get_vars, count_vars
from utils.replay_buffer import ReplayBuffer
from utils import logger
from utils.common import save_variables, load_variables, batch_norm
from .noise import NormalActionNoise, AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise

# put the batch_normal layer behind the activation layer
def base_network(x, activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_normal=True):

    batch_train = tf.constant(True, dtype=tf.bool)

    out = tf.cast(x, dtype=tf.float32) / 255.0
    
    out = layers.conv2d(out, 32, kernel_size=[8, 8], stride=4, padding='SAME', activation_fn=activation)
    if use_batch_normal:
        out = batch_norm(out, train=batch_train, name="conv0_bn")
    out = layers.conv2d(out, 64, kernel_size=[4, 4], stride=2, padding='SAME', activation_fn=activation)
    if use_batch_normal:
        out = batch_norm(out, train=batch_train, name="conv1_bn")
    out = layers.conv2d(out, 64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=activation)
    if use_batch_normal:
        out = batch_norm(out, train=batch_train, name="conv2_bn")
    reshape = tf.reshape(out, shape=[-1, out.get_shape()[1] * out.get_shape()[2] * 64])
    out = reshape
    return out

# put batch_normal before the last activation layer
def actor_network(observations, num_actions, activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_normal=True):
    batch_train = tf.constant(True, dtype=tf.bool)

    x = base_network(observations, activation=activation, output_activation=output_activation, use_batch_normal=use_batch_normal)
    x = layers.fully_connected(x, 128, activation_fn=None)
    # x = layers.layer_norm(x, scale=True, center=True, activation_fn=activation)
    x = batch_norm(x, train=batch_train, name="actor_fc0_bn")
    x = layers.fully_connected(x, num_actions, activation_fn=None)
    # x = layers.layer_norm(x, scale=True, center=True, activation_fn=output_activation)
    x = batch_norm(x, train=batch_train, name="actor_output_bn")
    x = output_activation(x)
    return x

def critic_network(observations, actions, activation=tf.nn.relu, output_activation=tf.nn.relu, use_batch_normal=True):
    

    x = base_network(observations, activation=activation, output_activation=output_activation, use_batch_normal=use_batch_normal)
    # print("action shape: ", actions.get_shape().as_list())
    actions_dense = layers.fully_connected(actions, 128, activation_fn=activation)
    x = tf.concat([x, actions_dense], axis=-1)  # this assumes observation and action can be concatenated
    x = layers.fully_connected(x, 128, activation_fn=activation)
    x = layers.fully_connected(x, 1, activation_fn=None)
    # x = output_activation(x)
    return x

def actor_critic_network(observations, actions, activation=tf.nn.relu, output_activation=tf.nn.tanh):

    num_actions = actions.shape.as_list()[-1]

    with tf.variable_scope('pi'):
        pi = actor_network(observations, num_actions, activation=activation, output_activation=output_activation)
        # print(pi.op.name, pi.get_shape().as_list())
    with tf.variable_scope('q'):
        q = critic_network(observations, actions, activation=activation, output_activation=output_activation)
        # print(q.op.name, q.get_shape().as_list())
    with tf.variable_scope('q', reuse=True):
        q_pi = critic_network(observations, pi, activation=activation, output_activation=output_activation)
        # print(q_pi.op.name, q_pi.get_shape().as_list())
    return pi, q, q_pi

"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(env,
         session=tf.get_default_session(),
         seed=0,
         use_action_noise=True,
         use_param_noise=False,
         use_non_zero_terminal_state = False,
         noise_std=0.2,
         replay_size=int(1e4),
         gamma=0.99,
         total_steps = 1000000,
         steps_per_epoch=5000,
         polyak=0.995,
         pi_lr=1e-3,
         q_lr=1e-3,
         batch_size=100,
         start_steps=10000,
         max_each_epoch_len=1000,
         save_freq=1,
         model_file_load_path=None,
         model_file_save_path=None
         ):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    train_env, test_env = env, env
    observation_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.shape
    observation_range = (train_env.observation_space.low, train_env.observation_space.high)
    action_range = (train_env.action_space.low, train_env.action_space.high)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    number_actions = train_env.action_space.shape[-1]
    assert (np.abs(train_env.action_space.low) == train_env.action_space.high).all()  # we assume symmetric actions.
    max_action = train_env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    # Inputs to computation graph
    obs0_ph = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
    obs1_ph = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
    done_ph = tf.placeholder(tf.float32, shape=(None, 1), name='done')
    rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
    actions_ph = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')

    # action_throttle_noise = None
    # action_steer_noise = None
    action_noise = None
    param_noise = None
    if use_action_noise:
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(number_actions),
                                                    sigma=float(noise_std) * np.ones(number_actions))
        # action_throttle_noise = OrnsteinUhlenbeckActionNoise(mu=np.ones(number_actions-1)*0.5,
        #                                             sigma=float(noise_std) * np.ones(number_actions-1))
        # action_steer_noise_0 = OrnsteinUhlenbeckActionNoise(mu=np.ones(number_actions-1)*(-0.9),
        #                                             sigma=float(noise_std) * np.ones(number_actions-1))
        # action_steer_noise_1 = OrnsteinUhlenbeckActionNoise(mu=np.ones(number_actions-1)*(0.9),
        #                                             sigma=float(noise_std) * np.ones(number_actions-1))
        # action_steer_noise_2 = OrnsteinUhlenbeckActionNoise(mu=np.zeros(number_actions-1),
        #                                             sigma=float(noise_std) * np.ones(number_actions-1))
    elif use_param_noise:
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(noise_std), desired_action_stddev=float(noise_std))
    else:
        logger.info("No specified noise......")

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q, q_pi = actor_critic_network(obs0_ph, actions_ph)

    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, _, q_pi_targ = actor_critic_network(obs1_ph, actions_ph)

    # Experience buffer
    replay_buffer = ReplayBuffer(replay_size)

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % var_counts)

    # Bellman backup for Q function
    if use_non_zero_terminal_state:
        target_q = tf.stop_gradient(rewards_ph + gamma * q_pi_targ)
    else:
        target_q = tf.stop_gradient(rewards_ph + gamma * (1 - done_ph) * q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    td_error = q - target_q
    from utils.common import huber_loss
    q_loss = tf.reduce_mean( huber_loss(td_error) )

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)

    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    session.run(tf.global_variables_initializer())
    # 如果存在模型文件则加载model
    if model_file_load_path is not None:
        if os.path.exists(model_file_load_path):
            load_variables(model_file_load_path)
            logger.log('Loaded model from {}'.format(model_file_load_path))
        else:
            logger.error("Cannont find your model file!")

    session.run(target_init)

    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': obs0_ph, 'a': actions_ph}, outputs={'pi': pi, 'q': q})

    each_epoch_rewards = 0   #每一epoch的累计reward
    each_epoch_length = 0    #每一epoch的总步数
    epoch_rewards = deque(maxlen=100)  #记录每一幕的累计回报
    epoch_steps = deque(maxlen=100)  #记录每一幕的累计步数
    epoch_actions = deque(maxlen=100)  #记录actor生成的action
    epoch_qs = deque(maxlen=100)     #记录所有的q值
    epoch_actor_losses = deque(maxlen=100)  #记录actor网络的损失值
    epoch_critic_losses = deque(maxlen=100)  #记录critic网络的损失值
    test_epoch_rewards = deque(maxlen=100)
    test_epoch_steps = deque(maxlen=100)
    saved_mean_reward = None

    start_time = time.time()
    # we need to reset the environment twice because of the bug of Carla
    observation = train_env.reset()
    observation = train_env.reset_env()

    # 获取actor网络动作
    AXIS_ROW = 0
    THROTTLE_INDEX = 0
    STEER_INDEX = 1
    def get_action(observation, apply_noise=True):

        actions = session.run(pi, feed_dict={obs0_ph: [observation]})
        print("pi generated from actor: ", actions)
        a = actions[0]
        epoch_actions.append(a)
        if action_noise is not None and apply_noise:
            # mean_action = np.mean(epoch_actions, axis=AXIS_ROW)
            # if mean_action[STEER_INDEX] < -0.5:
            #     action_steer_noise = action_steer_noise_1
            # elif mean_action[STEER_INDEX] > 0.5:
            #     action_steer_noise = action_steer_noise_0
            # else:
            #     action_steer_noise = action_steer_noise_2
            # noise_throttle = action_throttle_noise()
            # noise_steer = action_steer_noise()
            # print("throttle noise: ", noise_throttle)
            # print("steer noise: ", noise_steer)
            # a[THROTTLE_INDEX] += noise_throttle
            # a[STEER_INDEX] += noise_steer
            noise = action_noise()
            assert noise.shape == a.shape
            a += noise

        return np.clip(a, a_min=action_range[0], a_max=action_range[1])

    # 在test_env中测试agent
    def test_agent(n=10):
        o = test_env.reset()
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset_env(), 0, False, 0, 0
            while not (d or (ep_len == max_each_epoch_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, apply_noise=False))
                ep_ret += r
                ep_len += 1

            test_epoch_rewards.append(ep_ret)
            test_epoch_steps.append(ep_len)

    epochs = total_steps // steps_per_epoch

    # Main loop: collect experience in env and update/log each epoch
    for step in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if step > start_steps:
            action = get_action(observation, apply_noise=True)
        else:
            action = train_env.action_space.sample()
            # action = np.random.random(size=number_actions)
        # 记录每次的action
        print("action = ", action)

        # Step the env
        observation2, reward, done, _ = train_env.step(action)
        each_epoch_rewards += reward
        each_epoch_length += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if each_epoch_length == max_each_epoch_len else done

        # Store experience to replay buffer
        replay_buffer.add(observation, action, reward, observation2, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        observation = observation2

        # 在每次运动轨迹结束或者到达 steps == max_each_epoch_len时进行DDPG网络更新
        if done or (each_epoch_length == max_each_epoch_len):

            for _ in range(each_epoch_length):
                batch_obs0, batch_actions, batch_rewards, batch_obs1, batch_done = replay_buffer.sample(batch_size)
                feed_dict = {obs0_ph: batch_obs0,
                             obs1_ph: batch_obs1,
                             actions_ph: batch_actions,
                             rewards_ph: batch_rewards,
                             done_ph: batch_done.astype('float32')
                             }

                # Q-learning update
                Q_loss, Q_value, Q_pi, _ = session.run([q_loss, q, q_pi, train_q_op], feed_dict)
                # print("Q value: ", Q_value)
                # print("Q value of pi: ", Q_pi)
                epoch_critic_losses.append(Q_loss)
                epoch_qs.append(Q_value)

                # Policy update
                PI_loss, _, _ = session.run([pi_loss, train_pi_op, target_update], feed_dict)
                epoch_actor_losses.append(PI_loss)

            # 记录每个epoch的累计回报和总步数
            epoch_rewards.append(each_epoch_rewards)
            epoch_steps.append(each_epoch_length)
            each_epoch_rewards = 0
            each_epoch_length = 0

            observation = train_env.reset_env()

        # End of epoch wrap-up
        if step > 0 and step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent()
            
            mean_100ep_reward = np.mean(epoch_rewards)
            mean_100ep_steps = np.mean(epoch_steps)
            test_ep_reward = np.mean(test_epoch_rewards)
            test_ep_steps = np.mean(test_epoch_steps)
            mean_actor_loss = np.mean(epoch_actor_losses)
            mean_critic_loss = np.mean(epoch_critic_losses)
            mean_q_value = np.mean(epoch_qs)
            duration = time.time() - start_time

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                if (saved_mean_reward is None or test_ep_reward > saved_mean_reward):
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward, test_ep_reward))
                    save_variables(model_file_save_path)
                    saved_mean_reward = test_ep_reward

            # Log info about epoch
            logger.record_tabular("epoch", epoch)
            logger.record_tabular("mean_epoch_rewards", mean_100ep_reward)
            logger.record_tabular("mean_epoch_steps", mean_100ep_steps)
            logger.record_tabular("test_epoch_rewards", test_ep_reward)
            logger.record_tabular("test_epoch_steps", test_ep_steps)
            logger.record_tabular("mean_actor_loss", mean_actor_loss)
            logger.record_tabular("mean_critic_loss", mean_critic_loss)
            logger.record_tabular("mean_q_value", mean_q_value)
            logger.record_tabular("total_steps", step)
            logger.record_tabular('duration', duration)
            logger.dump_tabular()

    return get_action


