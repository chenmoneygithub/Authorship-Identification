#here
import tensorflow as tf
import utils.glove as glove
import numpy as np
#from proj_rnn_cell import RNNCell

DUMMY_PATH="utils/glove/glove_dummy.txt"
token_list=glove.loadWordTokens(DUMMY_PATH)
tokens={}
for i in range(len(token_list)):
    tokens[token_list[i]]=i
wordVectors=glove.loadWordVectors(tokens,DUMMY_PATH,6)
#print wordVectors
# let's say the sentence is "this is a file", labeled 1
# "this is dummy", labeled 2
sentences=np.array([[0,1,2,4],[0,1,3,0] ])
mask=np.array([[0 ,0, 0, 1],[0,0,1,0]])
labels=np.array([[1,0,0],[0,1,0]])


n_classes=3
embed_size=6
max_length=4
batch_size=1
lr=0.001
n_features=6
hidden_size=10


# start buiding model
#cell=RNNCell(n_features,hidden_size)

input_placeholder=tf.placeholder(tf.int32,[None,max_length])
labels_placeholder=tf.placeholder(tf.int32,[None,n_classes])
mask_placeholder=tf.placeholder(tf.bool,[None,max_length])

U=tf.Variable(np.random.rand(hidden_size, n_classes).astype(np.float32),tf.float32)

# feed dict
feed_dict={input_placeholder:sentences,labels_placeholder:labels,mask_placeholder:mask}

emb=tf.Variable(wordVectors,dtype=tf.float32)
x=tf.nn.embedding_lookup(emb,input_placeholder)

h = tf.zeros([tf.shape(x)[0], hidden_size],tf.float32)

preds=[]
W_h = tf.Variable(np.random.rand(hidden_size, hidden_size).astype(np.float32),tf.float32)
W_x = tf.Variable(np.random.rand(n_features, hidden_size).astype(np.float32),tf.float32)
b1 = tf.Variable(np.random.rand(hidden_size).astype(np.float32),tf.float32)

for i in range(max_length):
    if i >= 1:
        tf.get_variable_scope().reuse_variables()
    h=tf.nn.sigmoid(tf.matmul(h, W_h) + tf.matmul(x[:,i,:], W_x) + b1)
    p=tf.matmul(h,U)
    print 'p',tf.shape(p)
    preds.append(p)

preds=tf.pack(preds)
preds2=tf.reshape(preds,[-1,max_length,n_classes])
print tf.shape(preds2)
labels_to_loss=tf.tile(labels_placeholder,[max_length,1])
loss = tf.nn.softmax_cross_entropy_with_logits(preds2,labels_to_loss)
loss2=tf.boolean_mask(loss,mask_placeholder)
loss3=tf.reduce_mean(loss2)
train_op=tf.train.AdamOptimizer(lr).minimize(loss)
print tf.shape(loss3)
print tf.shape(mask_placeholder)
#loss = tf.reduce_mean(loss*mask_placeholder)


#x=tf.reshape(x,[-1,max_length,embed_size])
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
xx=sess.run(x,feed_dict=feed_dict)
print 'embedding',xx
print 'embedding shape',np.shape(xx)
pp=sess.run(preds,feed_dict=feed_dict)
print 'after pack',pp
pp2=sess.run(preds2,feed_dict=feed_dict)
print 'after reshape',pp2
lalo=sess.run(labels_to_loss,feed_dict=feed_dict)
print 'labels to loss',lalo
ll=sess.run(loss,feed_dict=feed_dict)
print 'after softmax loss',ll
ll2=sess.run(loss2,feed_dict=feed_dict)
print 'after boolean_mask loss',ll2
ll3=sess.run(loss3,feed_dict=feed_dict)
print 'final loss',ll3
