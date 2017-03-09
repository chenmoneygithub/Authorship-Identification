#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

#logger = logging.getLogger("hw3.q3.1")
#logger.setLevel(logging.DEBUG)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LSTMCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):

        (state_h, state_c)=state

        scope = scope or type(self).__name__

        with tf.variable_scope(scope):


            W_i = tf.get_variable('W_i',
                                  [self.state_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            U_i = tf.get_variable('U_i',
                                  [self.input_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            b_i = tf.get_variable('b_i',
                                  [self.state_size, ],
                                  initializer = tf.contrib.layers.xavier_initializer())

            W_f = tf.get_variable('W_f',
                                  [self.state_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            U_f = tf.get_variable('U_f',
                                  [self.input_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            b_f = tf.get_variable('b_f',
                                  [self.state_size, ],
                                  initializer = tf.contrib.layers.xavier_initializer())

            W_o = tf.get_variable('W_o',
                                  [self.state_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            U_o = tf.get_variable('U_o',
                                  [self.input_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable('b_o',
                                  [self.state_size, ],
                                  initializer = tf.contrib.layers.xavier_initializer())

            W_c = tf.get_variable('W_c',
                                  [self.state_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            U_c = tf.get_variable('U_c',
                                  [self.input_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            b_c = tf.get_variable('b_c',
                                  [self.state_size, ],
                                  initializer = tf.contrib.layers.xavier_initializer())


            i = tf.nn.sigmoid(tf.matmul(inputs, U_i) + tf.matmul(state_h, W_i) + b_i)
            f = tf.nn.sigmoid(tf.matmul(inputs, U_f) + tf.matmul(state_h, W_f) + b_f)
            o = tf.nn.sigmoid(tf.matmul(inputs, U_o) + tf.matmul(state_h, W_o) + b_o)

            c_hat = tf.nn.tanh(tf.matmul(inputs, U_c) + tf.matmul(state_h, W_c) + b_o)

            new_state = f * state_c + i * c_hat
            output = o * tf.nn.tanh(new_state)


        return output, new_state
