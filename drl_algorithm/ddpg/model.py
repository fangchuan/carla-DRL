"""
*********************************************************************************************************
*
*	模块名称 : ddpg_model模块
*	文件名称 : model.py
*	版    本 : V1.0.0
*   作    者 ：fc
*   日    期 ：2019/03/11
*	说    明 :
*             This file provide the actor and critic network model.
*
* \par Class List:
*    1.    Model();
*    2.    Actor(Model);
*    3.    Critic(Model);
*
*
*   修订记录：
	2019-03-12：   1.0.0      build;
	                         修改Critic, observation经过network的输出, 在输出上concatenate action_state, 然后再经过一层全连接层;
	                         param_noise好像还有问题;

    2019-03-15:   1.0.0      给Actor网络最后添加layer_norm;

*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""

import tensorflow as tf
from baselines.common.models import get_network_builder


class Model(object):
    def __init__(self, name, network='mlp', **network_kwargs):
        self.name = name
        self.network_builder = get_network_builder(network)(**network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.network_builder(obs)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.tanh(x)
        return x

COLUMN_DIM = 1
ROW_DIM = 1
class Critic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            x = self.network_builder(obs)
            # action_expanded = tf.expand_dims(action, axis=COLUMN_DIM, name="action_expanded_1")  #action_expanded.shape: (None, 1, 4)
            # action_tiled = tf.tile(action_expanded, [1, x.get_shape()[1], 1], name="action_tiled_1")  # action_tiled.shape: (None, 84,2)
            # action_tiled_expanded = tf.expand_dims(action_tiled, axis=ROW_DIM,
            #                                        name="action_expanded_2")  # action_tiled_expanded_2.shape: (None,1,84,2)
            #
            # x = tf.concat([x, action_tiled_expanded], axis=ROW_DIM)
            x = tf.concat([x, action], axis=-1) #  this assumes observation and action can be concatenated
            x = tf.layers.dense(x, 64, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
