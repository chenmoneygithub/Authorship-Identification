#here
import tensorflow as tf
import utils.glove as glove
import utils.data_util as data_util
import numpy as np
#from proj_rnn_cell import RNNCell


#print wordVectors
# let's say the sentence is "this is a file", labeled 1
# "this is dummy", labeled 2

sentences=np.array([[0,1,2,4],[0,1,3,0] ])
sentences=np.array([[0,1,2,4],[0,1,3,5] ])
mask=np.array([[0 ,0, 0, 1],[0,0,1,0]])
labels=np.array([[1,0,0],[0,1,0]])


n_classes=3
embed_size=6
max_length=4
batch_size=1
lr=0.001
n_features=6
hidden_size=10
DUMMY_PATH="utils/glove/glove_dummy.txt"

token_list=glove.loadWordTokens(DUMMY_PATH)
tokens={}
for i in range(len(token_list)):
    tokens[token_list[i]]=i
wordVectors=glove.loadWordVectors(tokens,DUMMY_PATH,embed_size)
token_list.append("cqian23th7zhangrao")
tokens["cqian23th7zhangrao"]=len(token_list)-1
print 'WV',np.shape(wordVectors)
wordVectors=np.append(wordVectors,[np.zeros(embed_size)],axis=0)
print 'WV',np.shape(wordVectors)

wordVectors2=data_util.load_embeddings(DUMMY_PATH,embed_size)

assert(wordVectors.all()==wordVectors2.all())


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

# run through rnn
for i in range(max_length):
    if i >= 1:
        tf.get_variable_scope().reuse_variables()
    h=tf.nn.sigmoid(tf.matmul(h, W_h) + tf.matmul(x[:,i,:], W_x) + b1)
    p=tf.matmul(h,U)
    print 'p',tf.shape(p)
    preds.append(p)

# prediction
preds=tf.pack(preds)
preds2=tf.reshape(preds,[-1,max_length,n_classes])

# these are for verification
preds3=tf.nn.softmax(preds2)
preds4=tf.log(preds3)

# loss calculation
labels_to_loss=tf.tile(labels_placeholder,[max_length,1])
labels_to_loss=tf.reshape(labels_to_loss,[-1,max_length,n_classes])
loss = tf.nn.softmax_cross_entropy_with_logits(preds2,labels_to_loss)
loss2=tf.boolean_mask(loss,mask_placeholder)
loss3=tf.reduce_mean(loss2)

# training op
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

# test implementation
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
xx=sess.run(x,feed_dict=feed_dict)
print 'embedding',xx
print 'embedding shape',np.shape(xx)
pp=sess.run(preds,feed_dict=feed_dict)
print 'preds after pack',pp
pp2=sess.run(preds2,feed_dict=feed_dict)
print 'preds after reshape',pp2


pp3=sess.run(preds3,feed_dict=feed_dict)
print 'preds after softmax',pp3


mask2=np.stack([mask for i in range(n_classes)] ,2)
pred6=np.sum(np.multiply(pp3,mask2),1)
print 'test batch_pred',pred6

pp4=sess.run(preds4,feed_dict=feed_dict)
print 'preds after log',pp4

lalo=sess.run(labels_to_loss,feed_dict=feed_dict)
print 'labels to loss',lalo.shape, lalo
ll=sess.run(loss,feed_dict=feed_dict)
print 'after softmax loss',ll.shape,ll
ll2=sess.run(loss2,feed_dict=feed_dict)
print 'after boolean_mask loss',ll2
ll3=sess.run(loss3,feed_dict=feed_dict)
print 'final loss',ll3
