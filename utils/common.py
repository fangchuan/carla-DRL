"""
*********************************************************************************************************
*
*	模块名称 : common模块
*	文件名称 : common.py
*	版    本 : V1.0.0
*   作    者 ：fc
*   日    期 ：2019/03/03
*	说    明 :
*             包含常用打印函数, ArgumentParser函数, gym Wrap类等:
*
* \par Method List:
*    1.    DEBUG_PRINT(*kwargs);
*    2.    common_arg_parser();
*    3.    NoopResetEnv(gym.Wrapper);
*    4.    FireResetEnv(gym.Wrapper);
*    5.    EpisodicLifeEnv(gym.Wrapper);
*    6.    MaxAndSkipEnv(gym.Wrapper);
*    7.    ClipRewardEnv(gym.RewardWrapper);
*    8.    WarpFrame(gym.ObservationWrapper);
*    9.    NormalizedEnv(gym.ObservationWrapper);
*    10.    save_variables(save_path, variables=None, sess=None);
*    11.    load_variables(load_path, variables=None, sess=None);
*    12.   batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None);
*
*   修订记录：
	2019-03-05：   1.0.0      build;

	2019-03-13：   1.0.0      添加gym wrap类;
	                          添加batch_norm(),使用tf.nn.batch_normalization();
	                          添加save_variables() 和load_variables();


*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""
import os
import gym
import joblib
import cv2
import numpy as np
import tensorflow as tf

from collections import deque
from argparse import ArgumentParser
from gym import spaces
from tensorflow.python.training.moving_averages import assign_moving_average

cv2.ocl.setUseOpenCL(False)

try:
    import const
except:
    from . import const

const.DEBUG = 1
# DEBUG_PRINT函数，去除print时只需要将const.DEBUG=0
def DEBUG_PRINT(*kwargs):
    if const.DEBUG:
        print(*kwargs)



def common_arg_parser():

    argparser = ArgumentParser()
    argparser.add_argument(
        '--num_timesteps',
        type=float,
        default=1e8,
        dest='total_steps_num',
        help='the total steps for training')
    argparser.add_argument(
        '--params-file',
        metavar='params_file',
        default='dqn_parameters.json',
        help='path to parameters file.Default=dqn_parameters.json')
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
        '--gpu-id',
        type=int,
        default=0,
        metavar="GPU_ID",
        help='GPU device ID to use. Default:0')
    argparser.add_argument(
        '--play',
        default=False,
        action='store_true',
        help='play the trained model. Default:False')

    return argparser

class NoopResetEnv(gym.Wrapper):
    '''
    在reset后随机走若干steps, 以保证每次reset 返回的observation不一样
    '''
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    '''
    reset 后 agent 必须执行某个step
    '''
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    '''
       skip 若干 frames, 并挑选这些frames中max_observation 和 total_reward 返回
    '''
    def __init__(self, env, skip=4, use_image_only_observation=True):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        if use_image_only_observation:
            self._obs_image_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        else:
            self._obs_image_buffer = np.zeros((2,)+env.observation_space.spaces[0].shape, dtype=np.uint8)
            self._obs_measurement_buffer = np.zeros(env.observation_space.spaces[1].shape, dtype=np.float32)

        self._skip       = skip
        self._use_image_only_obs = use_image_only_observation

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)

            if i == self._skip - 2:
                if self._use_image_only_obs:
                    self._obs_image_buffer[0] = obs
                else:
                    self._obs_image_buffer[0] = obs[0]

            if i == self._skip - 1:
                if self._use_image_only_obs:
                    self._obs_image_buffer[1] = obs
                else:
                    self._obs_image_buffer[1] = obs[0]
                    self._obs_measurement_buffer = obs[1]

            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_image_buffer.max(axis=0)
        if self._use_image_only_obs:
            observation = max_frame
        else:
            observation = (max_frame, self._obs_measurement_buffer)

        return observation, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def reset_env(self, **kwargs):
        return self.env.reset_env(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    '''
       将reward 统一裁剪为 -1, 0, +1,
    '''
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    '''
       裁剪 frames(images), 范围,存储格式,大小形状
    '''
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)
    
    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    '''
       不建议使用， 因为返回 float32 类型的observation
    '''
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class NormalizedEnv(gym.ObservationWrapper):
    '''
       observation 归一化, 返回 float32 类型 observation
    '''
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)
        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        obs = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return obs

def make_atari(env_id, timelimit=True):
    # XXX(john): remove timelimit argument after gym is upgraded to allow double wrapping
    env = gym.make(env_id)
    if not timelimit:
        env = env.env
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_carla(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    # if 'FIRE' in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def batch_norm(x, train, eps=1e-03, decay=0.99, affine=True, name=None):
    '''

    :param x: input tensor
    :param train: True/False, whether train or not
    :param eps:   epsilon cofficient used in divsion
    :param decay:
    :param affine:
    :param name:
    :return:
    '''
    with tf.variable_scope(name, default_name='BatchNorm2d', reuse=tf.AUTO_REUSE):
        params_shape = [x.shape[-1]]
        moving_mean = tf.get_variable('mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable('variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)

        def mean_var_with_update():
            axises = list(np.arange(len(x.shape) - 1))
            mean, variance = tf.nn.moments(x, axes=axises, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            print("bn beta name : ", beta.name)
            print("bn gamma name : ", gamma.name)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x

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

def get_vars(scope):
    '''
       获取命名空间scope内的变量
    :param scope:
    :return:
    '''
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    '''
          返回命名空间scope内变量的个数
    :param scope:
    :return:
    '''
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )