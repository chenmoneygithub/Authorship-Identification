#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
In this file, we define a model for authorship attribution
"""

import logging
import tensorflow as tf

from model import Model
from utils.Progbar import Progbar
from utils.read_minibatch import read_minibatch

logger = logging.getLogger("authorship-attribution")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class AttributionModel(Model):
    """
    Implements special functionality for authorship-attribution models.
    """
    def __init__(self, helper, config, report=None):
        self.helper = helper
        self.config = config
        self.report = report

    def prepare_data_minibatch(self, raw_data):
        """
        This function prepares data for our model
        In practice, it will turn raw data into appropriate format that can be fed into our model

        Args:
            raw_data: the original data
        Returns:

        """
        raise NotImplementedError("Abstract method prepare_data_minibatch(self, raw_data) must be implemented. ")

    def evaluate(self, pred_label, true_label):
        """

        This function is to evaluate the performance of our model
        The measure is: number of correct predictions / total number of samples

        Args:
            pred_label: A list of predicted label
            true_label: A list of the real label

        Returns:
            the prediction accuracy of the model
        """
        correct_num = 0
        for i in range(len(pred_label)):
            if(pred_label[i] == true_label[i]):
                correct_num += 1
        accuracy = correct_num * 1.0 / len(pred_label)
        return accuracy

    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        prog = Progbar(target = 1 + int(len(train_examples) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        print("")

        #logger.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        logger.info("Evaluating on development data")
        token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = entity_scores[-1]
        return f1