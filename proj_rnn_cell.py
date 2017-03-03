#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q2.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class RNNCell(tf.nn.rnn_cell.RNNCell):
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
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~6-10 lines)
            W_h = tf.get_variable('W_h',
                                  [self.state_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            W_x = tf.get_variable('W_x',
                                  [self.input_size, self.state_size],
                                  initializer = tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('b',
                                  [self.state_size, ],
                                  initializer = tf.contrib.layers.xavier_initializer())
            new_state = tf.nn.sigmoid(tf.matmul(state, W_h) + tf.matmul(inputs, W_x) + b1)
            ### END YOUR CODE ###
        output = new_state
        return output, new_state
