
import numpy as np
import json

def read_minibatch(data, batch_size, max_length, shuffle = True):
    """
    Args:
        data: the list of raw data, in the form [[author_index, [word1, word2, ...]]
                                                 [author_index, [word1, word2, ...]]
                                                 [author_index, [word1, word2, ...]]
                                                 ...
                                                                                   ]
        max_length: the fixed length of a sentence

        batch_size: the number of rows in each batch

        shuffle: determine whether shuffle the input data or not

    Returns:
        a list of minibatches, in the form [batch1, batch2, ...]
        each batch is made of three lists:
        batch[0]: feature list
        batch[1]: mask list
        batch[2]: label list(only one number in each row)
    """
    data_size = len(data)
    indices = np.arange(len(data))
    batch_list = []
    if shuffle is True:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, batch_size):
        minibatch_indices = indices[minibatch_start : minibatch_start + batch_size]
        selected_data = [data[i] for i in minibatch_indices]
        batch = process_to_minibatch(selected_data, max_length)
        batch_list.append(batch)
    return batch_list

def process_to_minibatch(data, max_length):
    with open('../../data/glove/tokenToIndex', 'r') as f:
        try:
            wordToIndex = json.load(f)
        # if the file is empty the ValueError will be thrown
        except ValueError:
            wordToIndex = {}

    batch = []

    feat_list = []
    mask_list = []
    label_list = []
    for i in range(len(data)):
        minifeat_list = []
        minimask_list = []
        minilabel_list = []
        for j in range(max_length):
            if(j == len(data[i][1]) - 1):
                minimask_list.append(True)
                minifeat_list.append(match_word_to_vector("cqian23th7zhangrao", wordToIndex))
                #minilabel_list.append(data[i][0])
                #minifeat_list.append([0])
            elif (j < len(data[i][1]) - 1):
                if(j == max_length - 1):
                    minimask_list.append(True)
                else:
                    minimask_list.append(False)
                minifeat_list.append(match_word_to_vector(data[i][1][j], wordToIndex))
                #minilabel_list.append(data[i][0])
                #minifeat_list.append([data[i][1][j]])
            else:
                minimask_list.append(False)
                minifeat_list.append(match_word_to_vector("cqian23th7zhangrao", wordToIndex))
                #minilabel_list.append(data[i][0])

        feat_list.append(minifeat_list)
        mask_list.append(minimask_list)
        label_list.append(data[i][0])

    batch.append(feat_list)
    batch.append(mask_list)
    batch.append(label_list)

    return batch


def process_word2num(auth_news_list, word_dict, glove, max_length):
    res = []
    pad_zero = np.zeros(50, ).tolist()
    for news_ind in range(len(auth_news_list)):
        cur_list = []
        mask_list = []
        total_sent = 0
        for sent_ind in range(max_length):
            sent_temp = np.zeros(50, )
            total = 0
            if sent_ind < len(auth_news_list[news_ind][1]):
                for word_ind in range(len(auth_news_list[news_ind][1][sent_ind])):
                    if word_dict.has_key(str.lower(auth_news_list[news_ind][1][sent_ind][word_ind])):
                        glove_ind = word_dict[str.lower(auth_news_list[news_ind][1][sent_ind][word_ind])]
                        sent_temp += glove[glove_ind]
                        total += 1
                if(total == 0):
                    cur_list.append(pad_zero)
                else:
                    sent_temp /= total
                    cur_list.append(sent_temp)
                mask_list.append(1)
                total_sent += 1
            else:
                cur_list.append(pad_zero)
                mask_list.append(0)
        mask_list = ( np.array(mask_list) * 1.0 / total_sent ).tolist()
        res.append([ auth_news_list[news_ind][0], cur_list, mask_list ])

    return res


def process_word2num_noglove(auth_news_list, word_dict, max_sent_num, max_sent_length):
    res = []
    pad_ind = word_dict["cqian23th7zhangrao"] # for unseen word
    for news_ind in range(len(auth_news_list)):
        sent_list = []
        sent_mask_list = []
        total_sent = 0
        for sent_ind in range(max_sent_num):
            word_list = []
            word_mask_list = []
            total_word = 0
            if sent_ind < len(auth_news_list[news_ind][1]):
                for word_ind in range(max_sent_length):
                    # the word not included in glove will not be considered!!
                    # Important! this part needs to be examined
                    if word_ind >= len(auth_news_list[news_ind][1][sent_ind]):
                        word_list.append(pad_ind)
                        word_mask_list.append(0)
                    else:
                        if word_dict.has_key(str.lower(auth_news_list[news_ind][1][sent_ind][word_ind])):
                            word_list.append(word_dict[str.lower(auth_news_list[news_ind][1][sent_ind][word_ind])])
                            word_mask_list.append(1)
                            total_word += 1
                        else:
                            word_list.append(pad_ind)
                            word_mask_list.append(0)
                if total_word != 0:
                    word_mask_list = (np.array(word_mask_list) * 1.0 / total_word).tolist()
                sent_mask_list.append(1)
                total_sent += 1
            else:
                word_list = (np.zeros([max_sent_length, ]) + pad_ind).tolist()
                word_mask_list = np.zeros([max_sent_length, ]).tolist()
                sent_mask_list.append(0)
            sent_list.append([word_list, word_mask_list])
        sent_mask_list = (np.array(sent_mask_list) * 1.0 / total_sent).tolist()
        res.append([auth_news_list[news_ind][0], sent_list, sent_mask_list])
    return res


def pack_batch_list(batch_list, batch_size):
    '''

    Args:
        batch_list:
            batch_list[0]: label
            batch_list[1]: feature
            batch_list[2]: mask
    Returns:
        packed batch list
    '''
    packed_list = []
    for start in range(0, len(batch_list), batch_size):
        label_batch = []
        feat_batch = []
        mask_batch = []
        for i in range(start, min(len(batch_list), start + batch_size)):
            label_batch.append(batch_list[i][0])
            feat_batch.append(batch_list[i][1])
            mask_batch.append(batch_list[i][2])
        packed_list.append([label_batch, feat_batch, mask_batch])
    return packed_list

def match_word_to_vector(word, word_dict):
    # this function is to map a word to its Glove vector
    if(str.lower(word) in word_dict):
        ind = word_dict[str.lower(word)]
        return ind
    else:
        return word_dict["cqian23th7zhangrao"]


def convertOnehotLabel(label_index_list, n_classes):
    label_array = np.zeros([len(label_index_list), n_classes])
    for i in range(len(label_index_list)):
        label_array[i][label_index_list[i]] = 1
    return label_array

if __name__ == "__main__":

    data = [ [1, ['a', 'a', 'a', 'a', 'a', 'a']],
             [0, [1, 2, 3, 4, 5, 6]],
             [1, [1, 2, 3, 4, 5, 6]],
             [3, [1, 2, 3, 4, 5, 6]],
             [1, [1, 2, 3, 4, 5, 6]],
             [1, [1, 2, 3, 4, 5, 6]],
             [7, [1, 2, 3, 4, 5, 6]],
             [1, [1, 2, 3, 4, 5, 6, 7, 8]],
             [19, [1, 2, 3, 4]],
             [1, [1, 2, 3, 4, 5, 6]],
             [1, [1, 2, 3, 4, 5, 6]],
             [1, [1, 2, 3, 4, 5, 6]],
             [1, [1, 2, 3, 4, 5, 6]]]

    batch_list = read_minibatch(data, 3, 6)
    print (batch_list)
