import glove
import numpy as np
import matplotlib.pyplot as plt

"""
data util functions goes here
"""


DEFAULT_FILE_PATH2 = "../../data/glove/glove.6B.50d.txt"
DEFAULT_DIMENSION = 50

"""
this function takes in two arguments
path
    is the path of the word vector file
d
    is the dimension of word vectors
(this might be redundant but I just want
to make sure we read in the correct embeddings)
"""
def load_embeddings(path=DEFAULT_FILE_PATH2,d=DEFAULT_DIMENSION):

    token_list=glove.loadWordTokens(path)
    tokens={}
    for i in range(len(token_list)):
        tokens[token_list[i]]=i
    embeddings=glove.loadWordVectors(tokens,path,d)

    #assert(embeddings.shape[1]==d,"Conflict between file and dimension")

    token_list.append("cqian23th7zhangrao")
    tokens["cqian23th7zhangrao"]=len(token_list)-1
    embeddings=np.append(embeddings,[np.zeros(d)],axis=0)

    return embeddings

"""
this function calcuates the F1 score of all classes
input: confusion_matrix: the confusion matrix,
                        nparray of size (n_classes * n_classes)
output: all_F1: the F1 score of all classes
                        nparray of size (n_classes)
"""
def calculate_F1_all(confusion_matrix):
    truepositive=confusion_matrix.diagonal()
    alltrue=np.sum(confusion_matrix,axis=1)
    allpositive=np.sum(confusion_matrix,axis=0)
    precisions=np.divide(truepositive.astype(np.float32),alltrue)
    recalls=np.divide(truepositive.astype(np.float32),allpositive)
    return (2*precisions*recalls)/(precisions+recalls)


"""
This function calculates F1 score as specified
inputs: confusion_maxtrix: the confusion matrix,
                        nparray of size (n_classes * n_classes)
        average:         : how the F1 score is averaged
                        'macro' - blind average
                        'micro' - weighted average based on true numbers
                        None    - no average
"""

def calculate_F1(confusion_matrix,average='macro'):
    all_F1=calculate_F1_all(confusion_matrix)
    if average=='macro':
        return np.mean(all_F1)
    elif average=='micro':
        alltrue=np.sum(confusion_matrix,axis=1)
        weights=alltrue/np.sum(alltrue)
        return np.sum(weights*all_F1)
    elif average==None:
        return all_F1


"""
this function takes a confusion matrix and display it
"""
def visualize_cm(cm):
    l=np.shape(cm)[0]
    plt.imshow(cm, interpolation='nearest',cmap='hot')
    plt.xticks(np.arange(0,l),range(l))
    plt.yticks(np.arange(0,l),range(l))
    plt.colorbar()
    plt.show()



"""
find the author based on prediction
inputs:
    preds: np array of shape (n_sentences,n_classes) after softmax
           (likely coming from the function predict_on_batch)

    method: method of calculating who the author is
            if method is blind, it will find the author with the maximum counts
            if method is smart, it will add the predictions and find the argmax

outputs:
    authod_id: an integer, the index of the author in n_classes

"""
def find_author(preds,method='blind'):
    d=np.shape(preds)
    if method=='blind':
        preds_sum=np.sum(preds,axis=0)
        classes=np.argmax(preds,axis=1)
        counts=np.bincount(classes,minlength=d[1])

        if count_ele(counts,np.amax(counts))==1:
            return np.argmax(counts)
        else:
            return find_author_smart(preds)
    else:
        return find_author_smart(preds)

def find_author_smart(preds):
    preds_sum=np.sum(preds,axis=0)
    return np.argmax(preds_sum)

def count_ele(arr,ele):
    cnt=0
    for arr_ele in np.nditer(arr):
        if arr_ele==ele:
            cnt=cnt+1
    return cnt
