
import numpy as np

def read_minibatch(data, batch_size, shuffle = True):
    data_size = len(data)
    indices = np.range(len(data))
    batch_list = []
    if shuffle is True:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, batch_size):
        minibatch_indices = indices[minibatch_start : minibatch_start + batch_size]
        selected_data = data[minibatch_indices]
        minibatch_list = process_to_minibatch(selected_data)
        batch_list.extend(minibatch_list)

def process_to_minibatch(data, max_length):
    batch_list = []
    for i in range(max_length):
        minifeat_list = []
        minilabel_list = []
        minimask_list = []
        for j in range(len(data)):
            if(i >= len(data[i])):
                minimask_list.append(False)
                minilabel_list.append(data[j][0])
                minifeat_list.append(match_word_to_vector(" "))
            else:
                minimask_list.append(True)
                minilabel_list.append(data[j][0])
                minifeat_list.append(match_word_to_vector(data[j][1][i]))
        minibatch = []
        minibatch.append(minifeat_list, minilabel_list, minimask_list)
        batch_list.append(minibatch)
    return batch_list


def match_word_to_vector(word, word_dict, glove_mat):
    # this function is to map a word to its Glove vector
    if(word_dict.has_key(word) is True):
        ind = word_dict[word]
        return glove_mat[ind, :] # return the corresponding vector
    else:
        return None