#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import file2dict as fdt
import read_minibatch as rmb
import read_minibatch_avemask as rmba
import os
import pickle
import numpy as np
import json

batch_size = 64
max_length = 70 # for 3 sentences


with open('../../data/glove/tokenToIndex', 'r') as f:
    try:
        wordToIndex = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        wordToIndex = {}


cwd = os.getcwd()
data_path = cwd + '/../dataset/C50/C50test'
#auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
auth_sentbundle_num = fdt.file2auth_sentbundle_num(data_path, 3)
ind = np.arange(len(auth_sentbundle_num))
np.random.shuffle(ind)
index = ind
raw_data = [auth_sentbundle_num[i] for i in index ]
#batch_list = rmb.read_minibatch(raw_data, batch_size, max_length, False)
batch_list = rmba.read_minibatch(raw_data, batch_size, max_length, False)

output = open('../../data/batch_data/C50/data_test_bundle_seq.pkl', 'wb')
pickle.dump(batch_list, output, -1)
output.close()

print "Success!"

#pkl_file = open('../../data/batch_data/data.pkl', 'rb')
#data1 = pickle.load(pkl_file)


