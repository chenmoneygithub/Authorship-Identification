#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
import os
import pickle

import file2dict as fdt
import utils.read_minibatch as rmb
import utils.data_util as data_util

from datetime import datetime

import tensorflow as tf
import numpy as np

from AttributionModel import AttributionModel
from proj_rnn_cell import RNNCell
from proj_gru_cell import GRUCell

# from util import print_sentence, write_conll, read_conll
# from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
# from ner_model import NERModel
# from defs import LBLS
# from q2_rnn_cell import RNNCell
# from q3_gru_cell import GRUCell
#
logger = logging.getLogger("RNN_Author_Attribution")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    cell_type="lstm" # either rnn, gru or lstm

    window_size = 0

    max_length = 70 # longest length of a sentence we will process
    n_classes = 50 # in total, we have 50 classes
    dropout = 0.9

    embed_size = 50

    hidden_size = 300
    batch_size = 64

    n_epochs = 41
    regularization = 0

    max_grad_norm = 10.0 # max gradients norm for clipping
    lr = 0.001 # learning rate

    def __init__(self, args):

        #self.cell = args.cell

        self.cell = GRUCell(Config.embed_size, Config.hidden_size)
        if "output_path" in args:
            # Where to save things.
            self.output_path = args.output_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format("RNN", datetime.now())

        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"

        #self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"

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
        self.input_placeholder = tf.placeholder(tf.int32, [None, Config.max_length])
        self.labels_placeholder = tf.placeholder(tf.int32, [None, Config.n_classes])
        self.mask_placeholder = tf.placeholder(tf.float32, [None, Config.max_length])
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

        embeddingTensor = tf.Variable(self.pretrained_embeddings,tf.float32)
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

        if Config.cell_type=="lstm":
            print "lstm"
            cell_state = tf.zeros([tf.shape(x)[0], Config.hidden_size])
            hidden_state = tf.zeros([tf.shape(x)[0], Config.hidden_size])
            init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
            cell = tf.nn.rnn_cell.BasicLSTMCell(Config.hidden_size, state_is_tuple=True)
            inputs_series=tf.split(1,Config.max_length,x)
            inputs_series = [tf.reshape(input, [-1, config.embed_size]) for input in inputs_series ]
            outputs, current_state = tf.nn.rnn(cell, inputs_series, init_state)


            self.U = tf.get_variable('U',
                                  [Config.hidden_size, Config.n_classes],
                                  initializer = tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable('b2',
                                  [Config.n_classes, ],
                                  initializer = tf.contrib.layers.xavier_initializer())
            h = tf.zeros([tf.shape(x)[0], Config.hidden_size])

            preds=[tf.matmul(o, self.U) + self.b2 for o in outputs]
            self.raw_preds=tf.pack(preds)
            preds=tf.reshape(tf.transpose(self.raw_preds, [1, 0, 2]),[-1,Config.max_length,Config.n_classes])
            return preds


        else:
            dropout_rate = self.dropout_placeholder


        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!

        #cell = RNNCell(Config.embed_size, Config.hidden_size)
        #cell = GRUCell(Config.embed_size, Config.hidden_size)

            preds = [] # Predicted output at each timestep should go here!


            # Use the cell defined below. For Q2, we will just be using the
            # RNNCell you defined, but for Q3, we will run this code again
            # with a GRU cell!
            if Config.cell_type=="rnn":
                cell = RNNCell(Config.embed_size, Config.hidden_size)
            elif Config.cell_type=="gru":
                cell = GRUCell(Config.embed_size, Config.hidden_size)
            else:
                assert False, "Cell type undefined"
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

                for time_step in range(Config.max_length):
                    if time_step >= 1:
                        tf.get_variable_scope().reuse_variables()
                    o, h = cell(x[:,time_step,:], h)

                    o_drop = tf.nn.dropout(o, dropout_rate)
                    preds.append(tf.matmul(o_drop, self.U) + self.b2)


            # Make sure to reshape @preds here.

            self.raw_preds=tf.pack(preds)
            preds=tf.reshape(tf.transpose(self.raw_preds, [1, 0, 2]),[-1,Config.max_length,Config.n_classes])
            return preds


    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """

        pred_mask=tf.reshape(self.mask_placeholder,[-1,Config.max_length,1])
        pred_mask=tf.tile(pred_mask,[1,1,Config.n_classes])
        pred_masked= preds * pred_mask
        pred_masked=tf.reduce_sum(pred_masked,axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits(pred_masked, self.labels_placeholder)

        loss = tf.reduce_mean(loss) + config.regularization * ( tf.nn.l2_loss(self.U) )
        if config.cell_type == "gru":
            with tf.variable_scope("RNN/cell", reuse= True):
                # add regularization

                loss += config.regularization * (tf.nn.l2_loss(tf.get_variable("W_r"))
                                                 + tf.nn.l2_loss(tf.get_variable("U_r"))
                                                 + tf.nn.l2_loss(tf.get_variable("W_z"))
                                                 + tf.nn.l2_loss(tf.get_variable("U_z"))
                                                 + tf.nn.l2_loss(tf.get_variable("W_o"))
                                                 + tf.nn.l2_loss(tf.get_variable("U_o")))
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
        _, loss, pred = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, max_length)
            mask_batch: np.ndarray of shape (n_samples, max_length)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
            (after softmax)
        """
        feed = self.create_feed_dict(inputs_batch,mask_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        pred_mask=sess.run(self.mask_placeholder,feed_dict=feed)
        pred_mask=np.reshape(pred_mask,(-1,Config.max_length,1))
        pred_mask=np.tile(pred_mask,(1,1,Config.n_classes))
        pred_masked=predictions*pred_mask
        pred_masked=np.sum(pred_masked,axis=1)
        return pred_masked


    def test_model(self, session, batch_list):
        print "Now, testing on the test set..."
        total = 0
        accuCount = 0
        for batch in batch_list:
            batch_feat = np.array(batch[0], dtype = np.int32)
            batch_mask = np.array(batch[1], dtype = np.float32)

            pred = self.predict_on_batch(session, batch_feat, batch_mask)
            accuCount += np.sum(np.argmax(pred,1) == batch[2])
            total += len(batch[2])
        accu = accuCount * 1.0 / total
        logger.info( ("Test accuracy %f" %(accu)) )

    def test_trainset_model(self, session, batch_list):
        print "Now, testing on the trainig set, notice this is only for debugging..."
        total = 0
        accuCount = 0
        for batch in batch_list:
            batch_feat = np.array(batch[0], dtype = np.int32)
            batch_mask = np.array(batch[1], dtype = np.float32)

            pred = self.predict_on_batch(session, batch_feat, batch_mask)
            accuCount += np.sum(np.argmax(pred,1) == batch[2])
            total += len(batch[2])
        accu = accuCount * 1.0 / total
        logger.info( ("Test accuracy on training set is: %f" %(accu)) )

    def process_model_output(self):

        pkl_file = open('../data/batch_data/C50/data_article_test.pkl', 'rb')
        batch_list = pickle.load(pkl_file)
        pkl_file.close()

        test_size = int(len(batch_list) / 1)
        training_batch = batch_list[0 : len(batch_list) - test_size]
        print test_size, len(batch_list)
        testing_batch = batch_list[len(batch_list) - test_size : len(batch_list)]

        saver = tf.train.Saver()
        with tf.Session() as session:
            #session.run(init)
            load_path = "results/RNN/20170319_032842/model.weights_10"
            saver.restore(session, load_path)

            print "Now, collecting the model outputs..."
            total = 0
            accuCount = 0

            for batch in testing_batch:
                batch_feat = np.array(batch[1], dtype = np.int32)
                batch_mask = np.array(batch[2], dtype = np.float32)

                preds = predict_on_batch(session, batch_feat, batch_mask)
                author_find = data_util.find_author(preds)
                if author_find == batch[0]:
                    accuCount += 1
                total += 1
            accuracy = accuCount / total
            print ("The testing accuracy is: %f"%(accuracy))

            return accuracy


    def train_model(self):

        if not os.path.exists(config.log_output):
            os.makedirs(os.path.dirname(config.log_output))
        handler = logging.FileHandler(config.log_output)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(handler)

        '''
        pkl_file = open('../data/batch_data/C50/data_bundle_seq.pkl', 'rb')

        batch_list = pickle.load(pkl_file)
        pkl_file.close()

        test_size = int(len(batch_list) / 10)
        training_batch = batch_list[0 : len(batch_list) - test_size]
        print test_size
        testing_train_batch = batch_list[test_size : 2 * test_size]
        testing_batch = batch_list[len(batch_list) - test_size : len(batch_list)]
        '''

        train_file = open('../data/batch_data/C50/data_train_bundle_seq.pkl', 'rb')
        training_batch = pickle.load(train_file)
        train_file.close()


        test_file = open('../data/batch_data/C50/data_test_bundle_seq.pkl', 'rb')
        testing_batch = pickle.load(test_file)
        test_file.close()

        testing_train_batch = training_batch[0 : int(len(training_batch) / 2)]

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            #load_path = "results/RNN/20170310_1022/model.weights_20"
            #saver.restore(session, load_path)

            #the following is a test for what in tensor
            batch = training_batch[0]
            batch_label = rmb.convertOnehotLabel(batch[2],  Config.n_classes)
            batch_feat = np.array(batch[0], dtype = np.int32)
            batch_mask = np.array(batch[1], dtype = np.float32)
            feed = self.create_feed_dict(batch_feat, labels_batch=batch_label, mask_batch=batch_mask,
                                     dropout=Config.dropout)
            _, loss, raw_pred, pred = session.run([self.train_op, self.loss, self.raw_preds, self.pred], feed_dict=feed)
            ##############

            for iterTime in range(Config.n_epochs):
                loss_list = []
                smallIter = 0

                for batch in training_batch:
                    batch_label = rmb.convertOnehotLabel(batch[2],  Config.n_classes)

                    batch_feat = np.array(batch[0], dtype = np.int32)
                    batch_mask = np.array(batch[1], dtype = np.float32)
                    loss = self.train_on_batch(session, batch_feat, batch_label, batch_mask)
                    loss_list.append(loss)
                    smallIter += 1

                    if(smallIter % 20 == 0):

                        #self.test_trainset_model(session, testing_train_batch)
                        #self.test_model(session, testing_batch)
                        logger.info(("Intermediate epoch %d Total Iteration %d: loss : %f" %(iterTime, smallIter, np.mean(np.mean(np.array(loss)))) ))
                self.test_trainset_model(session, testing_train_batch)
                self.test_model(session, testing_batch)
                if(iterTime % 10 == 0):
                    logger.info(("epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) ))
                    saver.save(session, self.config.model_output + "_%d"%(iterTime))

                    #if(smallIter % 200 == 0):
                    print ("Intermediate epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) )

            print ("epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) )


    def __init__(self, config, pretrained_embeddings, report=None):

        super(RNNModel, self).__init__(config)
        self.pretrained_embeddings = pretrained_embeddings

        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None
        self.raw_preds = None
        self.build()


if __name__ == "__main__":
    args = "gru"
    config = Config(args)
    glove_path = "../data/glove/glove.6B.50d.txt"
    glove_vector = data_util.load_embeddings(glove_path, config.embed_size)
    model = RNNModel(config, glove_vector.astype(np.float32))
    model.train_model()
