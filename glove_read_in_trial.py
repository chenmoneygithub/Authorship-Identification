
import numpy as np
import utils.glove as glove

# tokens is a dictionary that maps a token to its index
token_list=["this","is","a","dummy","file"]

# create an empty dictionary
tokens={}
for i in range(len(token_list)):
    tokens[token_list[i]]=i

# the path of the file where the word vectors are stored
DUMMY_PATH="utils/glove/glove_dummy.txt"

# read in word vectors
# the function takes 3 arguments:
"""
tokens:     a dictionary maps the token to their index in token_list
filepath:   a string, the path of the word vector file to be read
dimension:  integer, the length of the vector
            (it must be consistent with the file)
"""
dummy_vectors=glove.loadWordVectors(tokens,DUMMY_PATH,6)

for i in range(len(dummy_vectors)):
    # print the words (formatted to have a tab behind them) and the word vectors
    print "{0}\t".format(token_list[i]),dummy_vectors[i]
