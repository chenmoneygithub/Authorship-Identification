#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from AttributionModel import AttributionModel
from proj_rnn_cell import RNNCell

# from util import print_sentence, write_conll, read_conll
# from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
# from ner_model import NERModel
# from defs import LBLS
# from q2_rnn_cell import RNNCell
# from q3_gru_cell import GRUCell
#
# logger = logging.getLogger("hw3.q2")
# logger.setLevel(logging.DEBUG)
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    window_size = 0

    max_length = 35 # longest length of a sentence we will process
    n_classes = 51 # in total, we have 50 classes
    dropout = 0.5

    embed_size = 50

    hidden_size = 300
    batch_size = 64

    n_epochs = 10

    max_grad_norm = 10.0 # max gradients norm for clipping
    lr = 0.001 # learning rate

    def __init__(self, args):
        self.cell = args.cell

        if "output_path" in args:
            # Where to save things.
            self.output_path = args.output_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())

        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"

        #self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        #self.log_output = self.output_path + "log"

class RNNModel(AttributionModel):
    """
    Implements a recurrent neural network with an embedding layer and
    single hidden layer.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        """
        self.input_placeholder = tf.placeholder(tf.int32, [None, self.max_length])
        self.labels_placeholder = tf.placeholder(tf.int32, [None, self.n_classes])
        self.mask_placeholder = tf.placeholder(tf.bool, [None, self.max_length])
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        feed_dict = {}
        if labels_batch != None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch != None:
            feed_dict[self.input_placeholder] = inputs_batch
        if dropout != None:
            feed_dict[self.dropout_placeholder] = dropout
        if mask_batch != None:
            feed_dict[self.mask_placeholder] = mask_batch

        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """

        embeddingTensor = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embeddingTensor, self.input_placeholder)

        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = [] # Predicted output at each timestep should go here!

        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!
        cell = RNNCell(Config.n_features, Config.hidden_size)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.

        self.U = tf.get_variable('U',
                              [Config.hidden_size, Config.n_classes],
                              initializer = tf.contrib.layers.xavier_initializer())
        self.b2 = tf.get_variable('b2',
                              [Config.n_classes, ],
                              initializer = tf.contrib.layers.xavier_initializer())
        h = tf.zeros([tf.shape(x)[0], Config.hidden_size])

        with tf.variable_scope("RNN"):

            for time_step in range(self.max_length):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                o, h = cell(x[:,time_step,:], h)

                o_drop = tf.nn.dropout(o, dropout_rate)
                preds.append(tf.matmul(o_drop, self.U) + self.b2)

        # Make sure to reshape @preds here.
        #????? Do we need reshape?
        # preds = tf.pack(preds)
        # preds = tf.transpose(preds, perm = [1, 0, 2])
        # preds = tf.reshape(preds, [-1, Config.max_length, Config.n_classes])
        preds=tf.pack(preds)
        preds=tf.reshape(preds,[-1,Config.max_length,Config.n_classes])
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        labels_to_loss=tf.tile(labels_placeholder,[Config.max_length,1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, labels_to_loss)
        loss = tf.boolean_mask(loss,mask_placeholder)
        loss = tf.reduce_mean(loss)

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """

        train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)

        return train_op


    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):

        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, pretrained_embeddings, report=None):

        super(RNNModel, self).__init__(helper, config, report)
        self.pretrained_embeddings = pretrained_embeddings

        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()

if __name__ == "__main__":
