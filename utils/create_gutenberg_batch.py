#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import file2dict as fdt
import read_minibatch as rmb
import file2list3 as fdt3
import os
import pickle
import numpy as np
import json
from data_util import load_embeddings

batch_size = 16
max_sent_num = 15
max_sent_length = 30

cwd = os.getcwd()
data_path = cwd + '/../dataset/gutenberg'

with open('../../data/glove/tokenToIndex', 'r') as f:
    try:
        wordToIndex = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        wordToIndex = {}


#auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
#auth_sentbundle_num = fdt.file2auth_sentbundle_num(data_path, 3)[1:1000]

auth_news_num = fdt3.file2list(data_path)

ind = np.arange(len(auth_news_num))
np.random.shuffle(ind)
index = ind
raw_data = [auth_news_num[i] for i in index ]

batch_list = rmb.process_word2num_noglove(raw_data, wordToIndex, max_sent_num, max_sent_length)

batch_list_bundle = rmb.pack_batch_list(batch_list, batch_size)

output = open('../../data/batch_data/gutenberg/data_sentence_index.pkl', 'wb')
pickle.dump(batch_list_bundle, output, -1)
output.close()

print "Success!"

#print batch_list