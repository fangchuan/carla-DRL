import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import os
from baselines.common.schedules import LinearSchedule
from collections import deque
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.common import get_vars, count_vars
from utils import logger
from utils.common import save_variables, load_variables, batch_norm
from utils.noise import NormalActionNoise, AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise

def gaussian_likelihood(x, mu, log_std):
    EPS = 1e-8
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def apply_squashing_function(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


def base_network(x, activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_normal=True):
    batch_train = tf.constant(True, dtype=tf.bool)

    if isinstance(x, tuple):
        x_image = tf.cast(x[0], dtype=tf.float32) / 255.0
        # x_measure = batch_norm(x[1], train=batch_train, name="measurements_input_bn")
        x_measure = x[1]
    else:
        x_image = tf.cast(x, dtype=tf.float32) / 255.0
        x_measure = None
    # send images t CNN
    out_image = layers.conv2d(x_image, 32, kernel_size=[8, 8], stride=4, padding='SAME', activation_fn=activation)
    if use_batch_normal:
        out_image = batch_norm(out_image, train=batch_train, name="conv0_bn")
    out_image = layers.conv2d(out_image, 64, kernel_size=[4, 4], stride=2, padding='SAME', activation_fn=activation)
    if use_batch_normal:
        out_image = batch_norm(out_image, train=batch_train, name="conv1_bn")
    out_image = layers.conv2d(out_image, 64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=activation)
    if use_batch_normal:
        out_image = batch_norm(out_image, train=batch_train, name="conv2_bn")
    image_reshape = tf.reshape(out_image, shape=[-1, out_image.get_shape()[1] * out_image.get_shape()[2] * 64])
    # send measurements data to dense layer
    if x_measure is not None:
        out_measurement = layers.fully_connected(x_measure, 128, activation_fn=None)
        # out_measurement = batch_norm(out_measurement, train=batch_train, name="measurements_fc0_bn")
        out_measurement = activation(out_measurement)
        out = tf.concat([image_reshape, out_measurement], axis=-1)
    else:
        out = image_reshape

    return out



def actor_network(observations, num_actions, activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_normal=True):

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    batch_train = tf.constant(True, dtype=tf.bool)

    x = base_network(observations, activation=activation, output_activation=output_activation,
                     use_batch_normal=use_batch_normal)

    x_mu = layers.fully_connected(x, 64, activation_fn=activation)
    x_mu = batch_norm(x_mu, train=batch_train, name="actor_mu_fc0_bn")
    x_mu = activation(x_mu)
    mu = layers.fully_connected(x_mu, num_actions,activation_fn=None)  #此处特地没有使用output_activation

    x_logstd = layers.fully_connected(x, 64, activation_fn=None)
    x_logstd = batch_norm(x_logstd, train=batch_train, name="actor_logstd_fc0_bn")
    x_logstd = activation(x_logstd)
    log_std = layers.fully_connected(x_logstd, num_actions, activation_fn=output_activation)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(shape=tf.shape(mu))*std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    mu, pi, logp_pi = apply_squashing_function(mu, pi, logp_pi)
    return mu, pi, logp_pi


def critic_q_network(observations, actions, activation=tf.nn.relu, output_activation=tf.nn.relu, use_batch_normal=True):

    x = base_network(observations, activation=activation, output_activation=output_activation,
                     use_batch_normal=use_batch_normal)

    actions_dense = layers.fully_connected(actions, 64, activation_fn=activation)
    x = tf.concat([x, actions_dense], axis=-1)  # this assumes observation and action can be concatenated
    x = layers.fully_connected(x, 64, activation_fn=activation)
    x = layers.fully_connected(x, 1, activation_fn=None)
    # x = output_activation(x)
    return x

def critic_v_network(observations, activation=tf.nn.relu, output_activation=tf.nn.relu, use_batch_normal=True):

    x = base_network(observations, activation=activation, output_activation=output_activation,
                     use_batch_normal=use_batch_normal)

    x = layers.fully_connected(x, 64, activation_fn=activation)
    x = layers.fully_connected(x, 1, activation_fn=None)
    # x = output_activation(x)
    return x

def actor_critic_network(observations, actions, activation=tf.nn.relu, output_activation=tf.nn.tanh, action_space=None):
    num_actions = actions.shape.as_list()[-1]
    max_action = action_space[1]
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = actor_network(observations, num_actions, activation=activation, output_activation=output_activation)

    mu *= max_action
    pi *= max_action

    with tf.variable_scope('q1'):
        q1 = critic_q_network(observations, actions, activation=activation, output_activation=output_activation)
    with tf.variable_scope('q1', reuse=True):
        q1_pi = critic_q_network(observations, pi, activation=activation, output_activation=output_activation)

    with tf.variable_scope('q2'):
        q2 = critic_q_network(observations, actions, activation=activation, output_activation=output_activation)
    with tf.variable_scope('q2', reuse=True):
        q2_pi = critic_q_network(observations, pi, activation=activation, output_activation=output_activation)

    with tf.variable_scope('v'):
        v = critic_v_network(observations)

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v


"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac(env,
        seed=0,
        session=tf.get_default_session(),
        steps_per_epoch=5000,
        total_steps=1000000,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha=0.2,
        batch_size=64,
        start_steps=10000,
        save_freq=1,
        max_each_epoch_len=1000,
        model_file_load_path=None,
        model_file_save_path=None,
        use_prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        use_image_only_observations=False
        ):



    tf.set_random_seed(seed)
    np.random.seed(seed)

    train_env, test_env = env, env

    action_shape = train_env.action_space.shape
    action_range = (train_env.action_space.low, train_env.action_space.high)
    if use_image_only_observations:
        observation_image_shape = train_env.observation_space.shape
        observation_range = (train_env.observation_space.low, train_env.observation_space.high)
    else:
        observation_image_shape = train_env.observation_space.spaces[0].shape
        observation_measurement_shape = train_env.observation_space.spaces[1].shape
        observation_image_range = (train_env.observation_space.spaces[0].low, train_env.observation_space.spaces[0].high)
        observation_measurement_range = (train_env.observation_space.spaces[1].low, train_env.observation_space.spaces[1].high)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    number_actions = train_env.action_space.shape[-1]
    assert (np.abs(train_env.action_space.low) == train_env.action_space.high).all()  # we assume symmetric actions.
    max_action = train_env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    # Inputs to computation graph
    obs0_ph = tf.placeholder(tf.float32, shape=(None,) + observation_image_shape, name='obs0')
    obs1_ph = tf.placeholder(tf.float32, shape=(None,) + observation_image_shape, name='obs1')
    if not use_image_only_observations:
        measurements0_ph = tf.placeholder(tf.float32, shape=(None,) + observation_measurement_shape, name='measure0')
        measurements1_ph = tf.placeholder(tf.float32, shape=(None,) + observation_measurement_shape, name='measure1')

    done_ph = tf.placeholder(tf.float32, shape=(None, 1), name='done')
    rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
    actions_ph = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
    importance_weight_ph = tf.placeholder(tf.float32, shape=(None, 1), name="importanceWeight")

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic_network(obs0_ph, actions_ph, action_space = action_range)

    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, v_targ = actor_critic_network(obs1_ph, actions_ph, action_space = action_range)

    # Experience buffer
    if use_prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(size=replay_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_steps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(replay_size)

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(rewards_ph + gamma * (1 - done_ph) * v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
    td_error = v_backup - v
    v_loss = 0.5 * tf.reduce_mean(td_error ** 2)* importance_weight_ph
    value_loss = q1_loss + q2_loss + v_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, td_error,
                train_pi_op, train_value_op, target_update]

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
    each_epoch_rewards = 0  # 每一epoch的累计reward
    each_epoch_length = 0  # 每一epoch的总步数
    epoch_rewards = deque(maxlen=100)  # 记录每一幕的累计回报
    epoch_steps = deque(maxlen=100)  # 记录每一幕的累计步数
    epoch_actions = deque(maxlen=100)  # 记录actor生成的action
    epoch_q1_values = deque(maxlen=100)  # 记录所有的q1值
    epoch_q2_values = deque(maxlen=100)  # 记录所有的q2值
    epoch_v_values = deque(maxlen=100)  # 记录所有的v值
    epoch_actor_losses = deque(maxlen=100)  # 记录actor网络的损失值
    epoch_pi_logpro = deque(maxlen=100)  #记录从策略网络采样的动作的对数概率
    epoch_q1_losses = deque(maxlen=100)  # 记录q1网络的损失值
    epoch_q2_losses = deque(maxlen=100)  # 记录q2网络的损失值
    epoch_v_losses = deque(maxlen=100)   #记录v网络的损失
    test_epoch_rewards = deque(maxlen=100)
    test_epoch_steps = deque(maxlen=100)
    saved_mean_reward = None

    def get_action(observations, deterministic=False):
        act_op = mu if deterministic else pi
        if use_image_only_observations:
            actions = session.run(act_op, feed_dict={obs0_ph: [observations]})
        else:
            actions = session.run(act_op, feed_dict={obs0_ph: [observations[0]],
                                                 measurements0_ph: [observations[1]]})
        print("pi generated from actor: ", actions)
        return actions[0]

    def test_agent(n=10):
        global session, mu, pi, q1, q2, q1_pi, q2_pi
        o = test_env.reset()
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset_env(), 0, False, 0, 0
            while not (d or (ep_len == max_each_epoch_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, deterministic=True))
                ep_ret += r
                ep_len += 1

            test_epoch_rewards.append(ep_ret)
            test_epoch_steps.append(ep_len)

    episodes = total_steps // steps_per_epoch
    start_time = time.time()
    # we need to reset the environment twice because of the bug of Carla
    observation = train_env.reset()
    observation = train_env.reset_env()
    # Main loop: collect experience in env and update/log each epoch
    for step in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if step > start_steps:
            action = get_action(observation)
        else:
            action = train_env.action_space.sample()

        # Step the env
        observation1, reward, done, _ = train_env.step(action)
        each_epoch_rewards += reward
        each_epoch_length += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if each_epoch_length == max_each_epoch_len else done

        # Store experience to replay buffer
        replay_buffer.add(observation, action, reward, observation1, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        observation = observation1

        # 在每次运动轨迹结束或者到达 steps == max_each_epoch_len时进行DDPG网络更新
        if done or (each_epoch_length == max_each_epoch_len):
            for training_iteration in range(each_epoch_length):
                if not use_prioritized_replay:
                    batch_obs0, batch_actions, batch_rewards, batch_obs1, batch_done = replay_buffer.sample(batch_size)
                    if use_image_only_observations:
                        feed_dict = {obs0_ph: batch_obs0,
                                     obs1_ph: batch_obs1,
                                     actions_ph: batch_actions,
                                     rewards_ph: batch_rewards,
                                     done_ph: batch_done.astype('float32'),
                                     importance_weight_ph: np.ones_like(batch_rewards)
                                     }
                    else:
                        feed_obs0 = np.array(list(batch_obs0[:, 0]), copy=False)
                        feed_measurements0 = np.array(list(batch_obs0[:, 1]), copy=False)
                        feed_obs1 = np.array(list(batch_obs1[:, 0]), copy=False)
                        feed_measurements1 = np.array(list(batch_obs1[:, 1]), copy=False)
                        feed_dict = {obs0_ph: feed_obs0,
                                     obs1_ph: feed_obs1,
                                     measurements0_ph: feed_measurements0,
                                     measurements1_ph: feed_measurements1,
                                     actions_ph: batch_actions,
                                     rewards_ph: batch_rewards,
                                     done_ph: batch_done.astype('float32'),
                                     importance_weight_ph: np.ones_like(batch_rewards)
                                     }
                else:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(step))
                    (batch_obs0, batch_actions, batch_rewards, batch_obs1, batch_done, batch_weights,
                     batch_idxes) = experience
                    if use_image_only_observations:
                        feed_dict = {obs0_ph: batch_obs0,
                                     obs1_ph: batch_obs1,
                                     actions_ph: batch_actions,
                                     rewards_ph: batch_rewards,
                                     done_ph: batch_done.astype('float32'),
                                     importance_weight_ph: batch_weights
                                     }
                    else:
                        feed_obs0 = np.array(list(batch_obs0[:, 0]), copy=False)
                        feed_measurements0 = np.array(list(batch_obs0[:, 1]), copy=False)
                        feed_obs1 = np.array(list(batch_obs1[:, 0]), copy=False)
                        feed_measurements1 = np.array(list(batch_obs1[:, 1]), copy=False)
                        feed_dict = {obs0_ph: feed_obs0,
                                     obs1_ph: feed_obs1,
                                     measurements0_ph: feed_measurements0,
                                     measurements1_ph: feed_measurements1,
                                     actions_ph: batch_actions,
                                     rewards_ph: batch_rewards,
                                     done_ph: batch_done.astype('float32'),
                                     importance_weight_ph: batch_weights
                                     }

                outs = session.run(step_ops, feed_dict)
                PI_loss = outs[0]
                Q1_loss = outs[1]
                Q2_loss = outs[2]
                V_loss = outs[3]
                Q1_value = outs[4]
                Q2_value = outs[5]
                V_value = outs[6]
                LOGPRO_PI = outs[7]
                td_errors = outs[8]

                if use_prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

                epoch_q1_losses.append(Q1_loss)
                epoch_q1_values.append(Q1_value)
                epoch_q2_losses.append(Q2_loss)
                epoch_q2_values.append(Q2_value)
                epoch_v_losses.append(V_loss)
                epoch_v_values.append(V_value)
                epoch_actor_losses.append(PI_loss)
                epoch_pi_logpro.append(LOGPRO_PI)


            # 记录每个epoch的累计回报和总步数
            epoch_rewards.append(each_epoch_rewards)
            epoch_steps.append(each_epoch_length)
            each_epoch_rewards = 0
            each_epoch_length = 0

            observation, reward, done = train_env.reset_env(), 0, False

        # End of epoch wrap-up
        if step > 0 and step % steps_per_epoch == 0:
            episode = step // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent()

            duration = time.time() - start_time
            mean_100ep_reward = np.mean(epoch_rewards)
            mean_100ep_steps = np.mean(epoch_steps)
            mean_test_100ep_reward = np.mean(test_epoch_rewards)
            mean_teat_100ep_steps = np.mean(test_epoch_steps)
            mean_actor_loss = np.mean(epoch_actor_losses)
            mean_q1_loss = np.mean(epoch_q1_losses)
            mean_q1_value = np.mean(epoch_q1_values)
            mean_q2_loss = np.mean(epoch_q2_losses)
            mean_q2_value = np.mean(epoch_q2_values)
            mean_v_value = np.mean(epoch_v_values)
            mean_v_loss = np.mean(epoch_v_losses)
            mean_pi_logpro = np.mean(epoch_pi_logpro)

            # Save model
            if (episode % save_freq == 0) or (episode == episodes - 1):
               # if (saved_mean_reward is None or test_ep_reward > saved_mean_reward):
               logger.log(
                        "Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_test_100ep_reward))
               save_variables(model_file_save_path)
               saved_mean_reward = mean_test_100ep_reward

            # Log info about epoch
            logger.record_tabular('Episode', episode)
            logger.record_tabular("mean_ep_rewards", mean_100ep_reward)
            logger.record_tabular("mean_ep_steps", mean_100ep_steps)
            logger.record_tabular("test_ep_rewards", mean_test_100ep_reward)
            logger.record_tabular("test_ep_steps", mean_teat_100ep_steps)
            logger.record_tabular("mean_actor_loss", mean_actor_loss)
            logger.record_tabular("mean_q1_loss", mean_q1_loss)
            logger.record_tabular("mean_q1_value", mean_q1_value)
            logger.record_tabular("mean_q2_loss", mean_q2_loss)
            logger.record_tabular("mean_q2_value", mean_q2_value)
            logger.record_tabular("mean_v_loss", mean_v_loss)
            logger.record_tabular("mean_v_value", mean_v_value)
            logger.record_tabular("mean_pi_logpro",mean_pi_logpro)
            logger.record_tabular("total_steps", step)
            logger.record_tabular('duration', duration)
            logger.dump_tabular()

    return get_action
