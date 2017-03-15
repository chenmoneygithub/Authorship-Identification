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

logger = logging.getLogger("RNN_Author_Attribution")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    cell_type="gru" # either rnn, gru or lstm

    window_size = 0

    word_num = 30
    max_length = 15 # longest length of a sentence we will process
    n_classes = 50 # in total, we have 50 classes
    dropout = 0.9

    embed_size = 50

    hidden_size = 100
    batch_size = 16

    n_epochs = 1
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

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        """
        self.input_placeholder1 = tf.placeholder(tf.int32, [None, Config.max_length, Config.word_num])
        self.batch_mask_placeholder1 = tf.placeholder(tf.float32, [None, Config.max_length, Config.word_num])
        self.labels_placeholder1 = tf.placeholder(tf.int32, [None, Config.n_classes])
        self.mask_placeholder1 = tf.placeholder(tf.float32, [None, Config.max_length])

        self.input_placeholder2 = tf.placeholder(tf.int32, [None, Config.max_length, Config.word_num])
        self.batch_mask_placeholder2 = tf.placeholder(tf.float32, [None, Config.max_length, Config.word_num])
        self.labels_placeholder2 = tf.placeholder(tf.int32, [None, Config.n_classes])
        self.mask_placeholder2 = tf.placeholder(tf.float32, [None, Config.max_length])

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch1, batch_feat_mask1, mask_batch1,
                        inputs_batch2, batch_feat_mask2, mask_batch2,
                         labels_batch1=None, labels_batch2=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        feed_dict = {}
        feed_dict[self.batch_mask_placeholder1] = batch_feat_mask1
        if labels_batch1 != None:
            feed_dict[self.labels_placeholder1] = labels_batch1
        if inputs_batch1 != None:
            feed_dict[self.input_placeholder1] = inputs_batch1
        if mask_batch1 != None:
            feed_dict[self.mask_placeholder1] = mask_batch1

        feed_dict[self.batch_mask_placeholder2] = batch_feat_mask2
        if labels_batch1 != None:
            feed_dict[self.labels_placeholder2] = labels_batch2
        if inputs_batch1 != None:
            feed_dict[self.input_placeholder2] = inputs_batch2
        if mask_batch1 != None:
            feed_dict[self.mask_placeholder2] = mask_batch2

        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """

        embeddingTensor = tf.Variable(self.pretrained_embeddings, tf.float32)

        # For input 1
        embeddingsTemp1 = tf.nn.embedding_lookup(embeddingTensor, self.input_placeholder1)
        mask_batch1 = tf.reshape(self.batch_mask_placeholder1, [-1, config.max_length, config.word_num, 1])
        mask_batch1 = tf.tile(mask_batch1, [1, 1, 1, config.embed_size])
        embeddings1 = tf.multiply(embeddingsTemp1, mask_batch1)
        embeddings1 = tf.reduce_sum(embeddings1, axis = 2)

        # For input 2
        embeddingsTemp2 = tf.nn.embedding_lookup(embeddingTensor, self.input_placeholder2)
        mask_batch2 = tf.reshape(self.batch_mask_placeholder2, [-1, config.max_length, config.word_num, 1])
        mask_batch2 = tf.tile(mask_batch2, [1, 1, 1, config.embed_size])
        embeddings2 = tf.multiply(embeddingsTemp2, mask_batch2)
        embeddings2 = tf.reduce_sum(embeddings2, axis = 2)

        return embeddings1, embeddings2

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

        x1, x2 = self.add_embedding()
        #x = self.input_placeholder

        dropout_rate = self.dropout_placeholder

        self.raw_preds = [] # Predicted output at each timestep should go here!


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
        h = tf.zeros([tf.shape(x1)[0], Config.hidden_size])

        with tf.variable_scope("RNN"):
            # For input 1
            for time_step in range(config.max_length):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                o, h = cell(x1[:,time_step,:], h)

                o_drop1 = tf.nn.dropout(o, dropout_rate)
                self.raw_preds1.append(tf.matmul(o_drop1, self.U) + self.b2)

            for time_step in range(config.max_length):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                o, h = cell(x2[:,time_step,:], h)

                o_drop2 = tf.nn.dropout(o, dropout_rate)
                self.raw_preds2.append(tf.matmul(o_drop2, self.U) + self.b2)

        # Make sure to reshape @preds here.

        preds1=tf.pack(self.raw_preds1)
        preds1=tf.reshape(tf.transpose(preds1, [1, 0, 2]),[-1,Config.max_length,Config.n_classes])

        preds2=tf.pack(self.raw_preds2)
        preds2=tf.reshape(tf.transpose(preds2, [1, 0, 2]),[-1,Config.max_length,Config.n_classes])

        return preds1, preds2


    def add_loss_op(self, preds1, preds2):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """


        self.pred_mask1=tf.reshape(self.mask_placeholder1,[-1,Config.max_length,1])
        self.pred_mask1=tf.tile(self.pred_mask1,[1,1,Config.n_classes])

        self.pred_masked1=tf.multiply(preds1,self.pred_mask1)
        self.pred_masked1=tf.reduce_sum(self.pred_masked1,axis=1)

        self.pred_mask2=tf.reshape(self.mask_placeholder2,[-1,Config.max_length,1])
        self.pred_mask2=tf.tile(self.pred_mask2,[1,1,Config.n_classes])

        self.pred_masked2=tf.multiply(preds2,self.pred_mask2)
        self.pred_masked2=tf.reduce_sum(self.pred_masked2,axis=1)


        loss1 = tf.nn.softmax_cross_entropy_with_logits(self.pred_masked1, self.labels_placeholder1)
        loss2 = tf.nn.softmax_cross_entropy_with_logits(self.pred_masked2, self.labels_placeholder2)

        loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2) + config.regularization * ( tf.nn.l2_loss(self.U) )

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


    def train_on_batch(self, sess, inputs_batch, batch_feat_mask, labels_batch, mask_batch):

        feed = self.create_feed_dict(inputs_batch, batch_feat_mask, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss, pred, pred_mask = sess.run([self.train_op, self.loss, self.pred, self.pred_mask], feed_dict=feed)

        return loss

    def predict_on_batch(self, sess, inputs_batch, batch_feat_mask, mask_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, max_length)
            mask_batch: np.ndarray of shape (n_samples, max_length)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
            (after softmax)
        """
        feed = self.create_feed_dict(inputs_batch, batch_feat_mask, mask_batch)
        predictions = sess.run(tf.nn.softmax(self.pred), feed_dict=feed)
        mask2=np.stack([mask_batch for i in range(Config.n_classes)] ,2)
        pred2=np.sum(np.multiply(predictions,mask2),1)
        return pred2


        """
        record the history of the model
        this function will append the following content to the file (opened as f):

        ###
        training model:
        starting time:

        parameters：
            cell_type:
            embed_size:
            hidden_size:
            learning_rate:
            regularization:
            batch_size:

        training history:
        epoch       loss     train_accu      test_accu
        0           1.2      0.6             0.4
        ...

        """
    def record_history_init(self,f):
        f.write("###\n")
        f.write("training model: "+sys.argv[0]+"\n")
        f.write("starting time: "+str(datetime.now())+"\n")
        f.write("\n")
        f.write("parameters:\n")
        f.write("\tcell_type: "+Config.cell_type+"\n")
        f.write("\tembed_size: {}\n".format(Config.embed_size))
        f.write("\thidden_size: {}\n".format(Config.hidden_size))
        f.write("\tlearning_rate: {0:.4f}\n".format(Config.lr))
        f.write("\tregularization: {0:.5f}\n".format(Config.regularization))
        f.write("\tbatch_size: {}\n".format(Config.batch_size))
        f.write("\n")
        f.write("training history:\n")
        f.write("epoch \t\t loss \t\t train_accu \t\t test_accu\n")


        """
        write one line of training history
        """
    def record_history_accu(self,f,n_epoch,average_train_loss,train_accu,test_accu):
        f.write("{0:d} \t\t {1:.5f} \t\t {2:.5f} \t\t {3:.5f}\n".format(n_epoch,average_train_loss,train_accu,test_accu))


        """
        add two empty lines and close the file
        """
    def record_history_finish(self,f):
        f.write("\n")
        f.write("\n")
        f.close()

    def test_model(self, session, batch_list):
        print "Now, testing on the test set..."
        total = 0
        accuCount = 0
        for batch in batch_list:
            batch_feat = np.array(batch[1], dtype = np.int32)[:, :, 0, :]
            batch_feat_mask = np.array(batch[1], dtype = np.float32)[:, :, 1, :]
            batch_mask = np.array(batch[2], dtype = np.float32)

            pred = self.predict_on_batch(session, batch_feat, batch_feat_mask, batch_mask)
            accuCount += np.sum(np.argmax(pred,1) == batch[0])
            total += len(batch[0])
        accu = accuCount * 1.0 / total
        logger.info( ("Test accuracy %f" %(accu)) )
        return accu

    def test_trainset_model(self, session, batch_list):
        print "Now, testing on the trainig set, notice this is only for debugging..."
        total = 0
        accuCount = 0
        for batch in batch_list:
            batch_feat = np.array(batch[1], dtype = np.int32)[:, :, 0, :]
            batch_feat_mask = np.array(batch[1], dtype = np.float32)[:, :, 1, :]
            batch_mask = np.array(batch[2], dtype = np.float32)

            pred = self.predict_on_batch(session, batch_feat, batch_feat_mask, batch_mask)
            accuCount += np.sum(np.argmax(pred,1) == batch[0])
            total += len(batch[0])
        accu = accuCount * 1.0 / total
        logger.info( ("Test accuracy on training set is: %f" %(accu)) )
        return accu

    def train_model(self):
        # modify txt name from here
        training_history_txt_filename='results/training_history.txt'

        if not os.path.exists(config.log_output):
            os.makedirs(os.path.dirname(config.log_output))
        handler = logging.FileHandler(config.log_output)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(handler)

        pkl_file = open('../data/batch_data/gutenberg/data_sentence_index.pkl', 'rb')
        batch_list = pickle.load(pkl_file)
        pkl_file.close()

        # write training_history
        training_history_file = open(training_history_txt_filename,'a+')
        self.record_history_init(training_history_file)

        test_size = int(len(batch_list) / 10)
        training_batch = batch_list[0 : len(batch_list) - test_size]
        print test_size, len(batch_list)
        testing_train_batch = batch_list[test_size : 2 * test_size]
        testing_batch = batch_list[len(batch_list) - test_size : len(batch_list)]

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            #load_path = "results/RNN/20170310_1022/model.weights_20"
            #saver.restore(session, load_path)

            #the following is a test for what in tensor
            batch = training_batch[0]
            batch_label = rmb.convertOnehotLabel(batch[0],  Config.n_classes)
            batch_feat = np.array(batch[1], dtype = np.int32)[:, :, 0, :]
            batch_feat_mask = np.array(batch[1], dtype = np.float32)[:, :, 1, :]
            batch_mask = np.array(batch[2], dtype = np.float32)
            feed = self.create_feed_dict(batch_feat, batch_feat_mask, labels_batch=batch_label, mask_batch=batch_mask,
                                     dropout=Config.dropout)
            _, loss, raw_pred, pred = session.run([self.train_op, self.loss, self.raw_preds, self.pred], feed_dict=feed)
            ##############


            for iterTime in range(Config.n_epochs):
                loss_list = []
                smallIter = 0

                for batch in training_batch:
                    batch_label = rmb.convertOnehotLabel(batch[0],  Config.n_classes)
                    batch_feat = np.array(batch[1], dtype = np.int32)[:, :, 0, :]
                    batch_feat_mask = np.array(batch[1], dtype = np.float32)[:, :, 1, :]
                    batch_mask = np.array(batch[2], dtype = np.float32)
                    #print batch_mask
                    loss = self.train_on_batch(session, batch_feat,batch_feat_mask, batch_label, batch_mask)
                    loss_list.append(loss)
                    smallIter += 1

                    if(smallIter % 20 == 0):

                        #self.test_trainset_model(session, testing_train_batch)
                        #self.test_model(session, testing_batch)
                        logger.info(("Intermediate epoch %d Total Iteration %d: loss : %f" %(iterTime, smallIter, np.mean(np.mean(np.array(loss)))) ))

                # record training history on this epoch
                train_accu=self.test_trainset_model(session, testing_train_batch)
                test_accu=self.test_model(session, testing_batch)
                average_train_loss=np.mean(np.array(loss_list))
                self.record_history_accu(training_history_file,iterTime,average_train_loss,train_accu,test_accu)

                if(iterTime % 10 == 0):
                    logger.info(("epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) ))
                    saver.save(session, self.config.model_output + "_%d"%(iterTime))

                    #if(smallIter % 200 == 0):
                    print ("Intermediate epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) )

            print ("epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) )

            self.record_history_finish(training_history_file)



    def __init__(self, config, pretrained_embeddings, report=None):

        super(RNNModel, self).__init__(config)
        self.pretrained_embeddings = pretrained_embeddings

        self.input_placeholder = None
        self.labels_placeholder = None
        self.batch_mask_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()


if __name__ == "__main__":
    args = "gru"
    config = Config(args)
    glove_path = "../data/glove/glove.6B.50d.txt"
    glove_vector = data_util.load_embeddings(glove_path, config.embed_size)
    model = RNNModel(config, glove_vector.astype(np.float32))
    model.train_model()
