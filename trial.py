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
labels=np.array([1,2])


n_classes=3
embed_size=6
max_length=4
batch_size=1

n_features=6
hidden_size=10


# start buiding model
#cell=RNNCell(n_features,hidden_size)

input_placeholder=tf.placeholder(tf.int32,[None,max_length])
labels_placeholder=tf.placeholder(tf.int32,[None,])
mask_placeholder=tf.placeholder(tf.int32,[None,max_length])

U=tf.Variable(tf.ones([hidden_size,n_classes]),tf.float32)

# feed dict
feed_dict={input_placeholder:sentences,labels_placeholder:labels,mask_placeholder:mask}

emb=tf.Variable(wordVectors,dtype=tf.float32)
x=tf.nn.embedding_lookup(emb,input_placeholder)

h = tf.zeros([tf.shape(x)[0], hidden_size],tf.float32)

preds=[]
W_h = tf.Variable(tf.ones([hidden_size, hidden_size]),tf.float32)
W_x = tf.Variable(tf.ones([n_features, hidden_size]),tf.float32)
b1 = tf.Variable(tf.ones([hidden_size, ]),tf.float32)

for i in range(max_length):
    if i >= 1:
        tf.get_variable_scope().reuse_variables()
    h=tf.nn.sigmoid(tf.matmul(h, W_h) + tf.matmul(x[:,i,:], W_x) + b1)
    p=tf.nn.softmax(tf.matmul(h,U))
    print 'p',tf.shape(p)
    preds.append(p)

preds=tf.pack(preds)
print tf.shape(preds)
loss = tf.ones([1,])
print tf.shape(loss)
print tf.shape(mask_placeholder)
#loss = tf.reduce_mean(loss*mask_placeholder)


#x=tf.reshape(x,[-1,max_length,embed_size])
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
xx=sess.run(x,feed_dict=feed_dict)
ll=sess.run(loss,feed_dict=feed_dict)
print xx
print ll
print tf.shape(xx)
