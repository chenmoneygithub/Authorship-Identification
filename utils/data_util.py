import glove
import numpy as np

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
    embeddings=glove.loadWordVectors(tokens,DUMMY_PATH,embed_size)

    assert(embeddings.shape[1]==d,"Conflict between file and dimension")

    token_list.append("cqian23th7zhangrao")
    tokens["cqian23th7zhangrao"]=len(token_list)-1
    embeddings=np.append(embeddings,[np.zeros(d)],axis=0)

    return embeddings
