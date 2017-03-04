# this file is to save a dictionary


import json
import glove as glove

data = {}

token_list=glove.loadWordTokens("../../data/glove/glove.6B.50d.txt")
token_list.append("cqian23th7zhangrao")
tokens = {}
for i in range(len(token_list)):
    tokens[token_list[i]]=i

# save to file:
with open('../../data/glove/tokenToIndex', 'w') as f:
    json.dump(tokens, f)

#data = {}
# load from file:
with open('../../data/glove/tokenToIndex', 'r') as f:
    try:
        data = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        data = {}






