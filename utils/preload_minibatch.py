#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import file2dict as fdt
import read_minibatch as rmb
import os
import pickle
import numpy as np

batch_size = 64
max_length = 35

cwd = os.getcwd()
data_path = cwd + '/../dataset/C50/C50train'
auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
ind = np.arange(len(auth_sent_num))
np.random.shuffle(ind)
index = ind
raw_data = [auth_sent_num[i] for i in index ]
batch_list = rmb.read_minibatch(raw_data, batch_size, max_length)

output = open('../../data/batch_data/data.pkl', 'wb')
pickle.dump(batch_list, output, -1)
output.close()

print "Success!"

#pkl_file = open('../../data/batch_data/data.pkl', 'rb')
#data1 = pickle.load(pkl_file)


