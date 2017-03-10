#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import file2dict as fdt
import read_minibatch as rmb
import read_minibatch_avemask as rmba
import os
import pickle
import numpy as np

batch_size = 64
max_length = 100 # for 3 sentences

cwd = os.getcwd()
data_path = cwd + '/../dataset/C50/C50train'
#auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
auth_sentbundle_num = fdt.file2auth_sentbundle_num(data_path, 7)
ind = np.arange(len(auth_sentbundle_num))
np.random.shuffle(ind)
index = ind
raw_data = [auth_sentbundle_num[i] for i in index ]
#batch_list = rmb.read_minibatch(raw_data, batch_size, max_length)
batch_list = rmba.read_minibatch(raw_data, batch_size, max_length)

output = open('../../data/batch_data/data_bundle_seqmask.pkl', 'wb')
pickle.dump(batch_list, output, -1)
output.close()

print "Success!"

#pkl_file = open('../../data/batch_data/data.pkl', 'rb')
#data1 = pickle.load(pkl_file)


