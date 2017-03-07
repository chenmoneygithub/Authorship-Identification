import file2dict as fdt
import utils.read_minibatch as rmb
import os
import pickle


batch_size = 64
max_length = 35

cwd = os.getcwd()
data_path = cwd + '/../dataset/C50/C50train'
auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
batch_list = rmb.read_minibatch(auth_sent_num, batch_size, max_length)

output = open('../../data/batch_data/data.pkl', 'wb')
pickle.dump(batch_list, output, -1)
output.close()

print "Success!"

#pkl_file = open('../../data/batch_data/data.pkl', 'rb')
#data1 = pickle.load(pkl_file)


