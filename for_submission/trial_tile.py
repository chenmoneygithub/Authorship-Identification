import tensorflow as tf
import numpy as np
import pickle
import utils.read_minibatch as rmb

class Config:

    max_length = 35 # longest length of a sentence we will process
    n_classes = 51 # in total, we have 50 classes
    dropout = 0.9

    embed_size = 50

    hidden_size = 300
    batch_size = 64

    n_epochs = 41
    regularization = 0

    max_grad_norm = 10.0 # max gradients norm for clipping
    lr = 0.001 # learning rate



pkl_file = open('../data/batch_data/data.pkl', 'rb')

batch_list = pickle.load(pkl_file)
pkl_file.close()

test_size = int(len(batch_list) / 10)
training_batch = batch_list[0 : len(batch_list) - test_size]
print test_size
testing_train_batch = batch_list[test_size : 2 * test_size]
testing_batch = batch_list[len(batch_list) - test_size : len(batch_list)]


input_placeholder = tf.placeholder(tf.int32, [None, Config.max_length])
labels_placeholder = tf.placeholder(tf.int32, [None, Config.n_classes])
mask_placeholder = tf.placeholder(tf.float32, [None, Config.max_length])

pred_mask=tf.reshape(mask_placeholder,[-1,Config.max_length,1])
pred_mask=tf.tile(pred_mask,[1,1,Config.n_classes])

preds=tf.pack(pred_mask)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for batch in training_batch:
    batch_label = rmb.convertOnehotLabel(batch[2],  Config.n_classes)

    batch_feat = np.array(batch[0], dtype = np.float32)
    batch_mask = np.array(batch[1], dtype = np.float32)
    feed_dict = {}

    feed_dict[labels_placeholder] = batch_label

    feed_dict[input_placeholder] = batch_feat

    feed_dict[mask_placeholder] = batch_mask

    bb, pred = sess.run([pred_mask, preds], feed_dict = feed_dict)
    print bb
