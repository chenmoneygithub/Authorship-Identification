
import numpy as np

DEFAULT_FILE_PATH = "utils/glove/glove.6B.50d.txt"

def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    wordVectors = np.zeros((len(tokens), dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data)
    return wordVectors

def loadWordTokens(filepath=DEFAULT_FILE_PATH):
    token_list=[]
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            token_list.append(token)
    return token_list
