#here
<<<<<<< Updated upstream
import random
import tensorflow as tf
=======
#import tensorflow as tf
>>>>>>> Stashed changes
import utils.glove as glove
#import utils.data_util as data_util
import numpy as np
import pickle
import utils.confusion_matrix as cm
import utils.data_util as du
#from proj_rnn_cell import RNNCell
import math
import sys

#print wordVectors
# let's say the sentence is "this is a file", labeled 1
# "this is dummy", labeled 2
def trial1():
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

def trial2():
    sentences=np.array([[0,1,2,4],[0,1,3,0] ])
    sentences=np.array([[0,1,2,4],[0,1,3,5] ])
    mask=np.array([[0 ,0, 0, 1],[0,0,1,0]])
    mask2=np.array([[0 ,0, 1, 1],[0,0,1,0]])
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
    feed_dict2={input_placeholder:sentences,labels_placeholder:labels,mask_placeholder:mask2}
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
    print np.shape(ll2)
    ll2=sess.run(loss2,feed_dict=feed_dict2)
    print 'after boolean_mask loss',ll2
    print np.shape(ll2)
    ll3=sess.run(loss3,feed_dict=feed_dict)
    print 'final loss',ll3


def trial3():
    a=np.array([[0,0,1,0],[0,1,0,0]])
    aa=np.random.rand(2,4,3)
    inputs=tf.placeholder(tf.float32,[2,4])
    inputs2=tf.placeholder(tf.float32,[2,4,3])
    feed_dict={inputs:a,inputs2:aa}
    b=tf.reshape(inputs,[2,4,1])
    b=tf.tile(b,[1,1,3])
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    bb=sess.run(b,feed_dict=feed_dict)
    print bb
    b2=sess.run(inputs2,feed_dict=feed_dict)
    print b2
    b3=sess.run(b*inputs2,feed_dict=feed_dict)
    print b3
    b4=sess.run(tf.reduce_sum(b*inputs2,1),feed_dict=feed_dict)
    print b4

def trial4():
    pkl_file = open('../data/batch_data/bbc/data_bundle.pkl', 'rb')

    batch_list = pickle.load(pkl_file)
    print batch_list
    pkl_file.close()

def trial5():
    real = [s for s in xrange(50)] * 10
    real.extend([random.randint(0, 49) for r in xrange(1000)])
    pred = [p for p in xrange(50)] * 10
    pred.extend([random.randint(0, 49) for r in xrange(1000)])

    t_cm = cm.generate_cm(real, pred, 50)
    x = t_cm.as_matrix().astype(np.uint8)
    du.visualize_cm(x)
    print "done"


# testing writing txt file
def trial6():
    f=open('results/writing_txt_trial.txt','a+')
    for i in range(2):
        f.write("###\n")
        f.write("Appended line %d\r\n" % (i+1))
        f.write("writing the value of pi: {0:.5f}\n".format(math.pi))
        f.write(sys.argv[0]+"\n")
        f.write("\tadditional \t \tpadding{}\n".format(3))
        f.write("\n")
    f.close()


def trial7():
    return 1

def main():

    trial5()

if __name__=="__main__":
    main()
