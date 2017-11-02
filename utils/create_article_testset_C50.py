#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import file2dict as fdt
import read_minibatch as rmb
import read_minibatch_avemask as rmba
import os
import pickle
import numpy as np
import json
from data_util import load_embeddings


max_sent_num = 3
max_length = 70

cwd = os.getcwd()
data_path = cwd + '/../dataset/C50/C50test'

with open('../../data/glove/tokenToIndex', 'r') as f:
    try:
        wordToIndex = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        wordToIndex = {}


#auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
#auth_sentbundle_num = fdt.file2auth_sentbundle_num(data_path, 3)[1:1000]

auth_news_num = fdt.file2auth_news_num(data_path)

ind = np.arange(len(auth_news_num))
np.random.shuffle(ind)
index = ind
raw_data = [auth_news_num[i] for i in index ]

batch_list = rmb.parse_article_sentbundle(raw_data, wordToIndex, max_sent_num, max_length)


output = open('../../data/batch_data/C50/data_article_test.pkl', 'wb')
pickle.dump(batch_list, output, -1)
output.close()

print "Success!"

#print batch_list