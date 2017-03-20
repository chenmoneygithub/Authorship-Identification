#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import file2dict as fdt
import read_minibatch as rmb
import file2dict as fdt
import os
import pickle
import numpy as np
import json
from data_util import load_embeddings

batch_size = 16
max_sent_num = 15
max_sent_length = 30

cwd = os.getcwd()
data_path = cwd + '/../dataset/C50/C50train'

with open('../../data/glove/tokenToIndex', 'r') as f:
    try:
        wordToIndex = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        wordToIndex = {}


#auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
#auth_sentbundle_num = fdt.file2auth_sentbundle_num(data_path, 3)[1:1000]

pair_list = fdt.file2pair(data_path, sample_num=1000)

ind = np.arange(len(pair_list))
np.random.shuffle(ind)
index = ind
raw_data = [pair_list[i] for i in index ]

batch_list = []
for article_ind in range(len(raw_data)):
    batch_list.append(rmb.process_word2num_noglove(raw_data[article_ind], wordToIndex, max_sent_num, max_sent_length))

pair_list_bundle = rmb.pack_pair_list(batch_list, batch_size)

output = open('../../data/batch_data/C50/data_sentence_pair.pkl', 'wb')
pickle.dump(pair_list_bundle, output, -1)
output.close()

print "Success!"

#print batch_list