
import numpy as np
import utils.glove as glove

"""
option 1: read in word vectors according to the order of our own token_list.

VS

option 2: read in word vectors according to the order inside the word vector file.
          i.e. the token_list are the words inside the word vector file
"""

# the path of the file where the word vectors are stored
DUMMY_PATH="utils/glove/glove_dummy.txt"

option =2
# token_list is the list containing all the tokens
# tokens is a dictionary that maps a token to its index
if option==1:
    token_list=["is","this","a","file","dummy"]
elif option ==2:
    token_list=glove.loadWordTokens(DUMMY_PATH)
else:
    assert false, 'Not a valid option'
# create an empty dictionary
tokens={}
for i in range(len(token_list)):
    tokens[token_list[i]]=i


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
