import os
import re

"""
input: data set directory
output: dict{author: [list of news files]}
"""
def read_mydir(input_dir):
    authors = os.listdir(input_dir)             # list the sub-dir of curr-dir
    author_newss_dict = {}
    for author in authors:                      # traverse each sub-directory
        if author[0] == '.':
            pass
        else:
            newss = os.listdir(input_dir + '/' + author)
            news_list = []
            for news in newss:                  # traverse each txt file in sub-dir
                news_dir = input_dir + '/' + author + '/' + news
                news_list.append(news_dir)
            author_newss_dict[author] = news_list
    return author_newss_dict



"""
input: single news path
output: list of sentences, each sentence is also a list of words
"""
def read_myfile(path):
    myfile = path
    f = open(myfile, 'r')

    f_list = []
    regex = re.compile(r"[.?!;][\'\"]?\s")
    for paragraph in f:
        paragraph = paragraph.strip()[0:-1]         # remove the last char in a line
        paragraph = re.sub(pattern = regex, repl = r". ", string = paragraph)
        sentences = paragraph.strip().split(". ")   # split line to sentences
        for sentence in sentences:
            words_list = []
            words = sentence.strip().split(" ")     # split sentence to words
            for word in words:
                word = word.strip(" ,)(][{}#$%\"\'").lower()         # strip a word
                words_list.append(word)
            if len(words_list) >= 1:                # at least 1 word
                f_list.append(words_list)

    return f_list



"""
input: data set directory
output: dict{author: [list of news]}
        each news is a list of sentences
        each sentence is a list of words
"""
def extract_name(path):
    portfolio = {}
    author_newss_dict = read_mydir(path)            # obtain {author:[file_list]} dict
    for author, newss in author_newss_dict.items(): # traverse all items in dict
        news_vec_list = []
        for news_dir in newss:                      # traverse each file in file_list
            news_vec = read_myfile(news_dir)        # parse file to sentence list
            real_news_vec = []
            for vec in news_vec:
                if len(vec) > 1:
                    real_news_vec.append(vec)
            news_vec_list.append(real_news_vec)
        portfolio[author] = news_vec_list
    return portfolio



"""
input: data set directory
output: dict{author_index: [list of news]}
        each news is a list of sentences
        each sentence is a list of words
"""
def extract_num(path):
    portfolio = {}
    idx = 0
    author_newss_dict = read_mydir(path)            # obtain {author:[file_list]} dict
    for author in author_newss_dict:                # traverse all items in dict
        newss_dir = author_newss_dict[author]
        news_vec_list = []
        for news_dir in newss_dir:                  # traverse each file in file_list
            news_vec = read_myfile(news_dir)        # parse file to sentence list
            real_news_vec = []
            for vec in news_vec:
                if len(vec) > 1:
                    real_news_vec.append(vec)
            news_vec_list.append(real_news_vec)
        portfolio[idx] = news_vec_list
        idx += 1
    return portfolio

"""
input: author-newss dict
output: author-news list
"""
def dict2auth_news_name(output_name):
    auth_news_name = []
    for auth in output_name:
        newss = output_name[auth]
        for news in newss:
            temp_list = []
            temp_list.append(auth)
            temp_list.append(news)
            auth_news_name.append(temp_list)
    return auth_news_name

"""
input: author-newss dict
output: author-news list
"""
def dict2auth_news_num(output_num):
    auth_news_num = []
    for auth in output_num:
        newss = output_num[auth]
        for news in newss:
            temp_list = []
            temp_list.append(auth)
            temp_list.append(news)
            auth_news_num.append(temp_list)
    return auth_news_num

"""
intput: author-newss dict
output: author-sent list
"""
def dict2auth_sent_name(output_name):
    auth_sent_name = []
    for auth in output_name:
        newss = output_name[auth]
        for news in newss:
            for sent in news:
                temp_list = []
                temp_list.append(auth)
                temp_list.append(sent)
                auth_sent_name.append(temp_list)
    return auth_sent_name

"""
input: author-newss dict
output: author-sent list
"""
def dict2auth_sent_num(output_num):
    auth_sent_num = []
    for auth in output_num:
        newss = output_num[auth]
        for news in newss:
            for sent in news:
                temp_list = []
                temp_list.append(auth)
                temp_list.append(sent)
                auth_sent_num.append(temp_list)
    return auth_sent_num

"""
input: dir path
output: author-news list
"""
def file2auth_news_name(path):
    output_name = extract_name(path)
    auth_news_name = dict2auth_news_name(output_name)
    return auth_news_name

"""
input: dir path
output: author-sent list
"""
def file2auth_sent_name(path):
    output_name = extract_name(path)
    auth_sent_name = dict2auth_sent_name(output_name)
  #  auth_sent_name = removeZeroLen(auth_sent_name)
    return auth_sent_name

"""
input: dir path
output: author-news list
"""
def file2auth_news_num(path):
    output_num = extract_num(path)
    auth_news_num = dict2auth_news_num(output_num)
    return auth_news_num

"""
input: dir path
output: author-sent list
"""
def file2auth_sent_num(path):
    output_num = extract_num(path)
    auth_sent_num = dict2auth_sent_num(output_num)
   # auth_sent_num = removeZeroLen(auth_sent_num)
    return auth_sent_num

"""
input: dir path
output: author-index dict
"""
def name2idx(path):
    dict = {}
    idx = 0
    author_newss_dict = read_mydir(path)            # obtain {author:[file_list]} dict
    for author in author_newss_dict:
        dict[author] = idx
        idx += 1
    return dict


def file2auth_sentbundle_num(path, sentence_num):
    output_num = extract_num(path)
    auth_news_num = dict2auth_news_num(output_num)
    bundle_sentence = []
    for news_ind in range(len(auth_news_num)):
        for start in range(0, len(auth_news_num[news_ind][1]), 2):
            temp_bundle = []
            for i in range(min(sentence_num, len(auth_news_num[news_ind][1]) - start )):
                temp_bundle.extend(auth_news_num[news_ind][1][start + i])
            bundle_sentence.append([auth_news_num[news_ind][0], temp_bundle ])
    return bundle_sentence

def removeZeroLen(list_sent):
    res = []
    for i in range(len(list_sent)):
        if len(list_sent[i][1]) > 1:
            res.append(list_sent[i])
    return res

"""
following is used for test
"""
if __name__=='__main__':
    '''
    from timeit import Timer
    t1=Timer("extract_name(\"./dataset/C50/C50train\")","from __main__ import extract_name")
    t2=Timer("extract_num(\"./dataset/C50/C50train\")","from __main__ import extract_num")
    print t1.timeit(1)
    print t2.timeit(1)
    '''

    cwd = os.getcwd()
    test_path = cwd + '/../dataset/bbc'


    auth_sentbundle_num = file2auth_sentbundle_num(test_path, 3)
    for x in auth_sentbundle_num:
        print x
        break

    auth_news_name = file2auth_news_name(test_path)
    for x in auth_news_name:
        print x
        break

    auth_sent_name = file2auth_sent_name(test_path)
    for x in auth_sent_name:
        print x
        break

    auth_news_num = file2auth_news_num(test_path)
    for x in auth_news_num:
        print x
        break

    auth_sent_num = file2auth_sent_num(test_path)
    for x in auth_sent_num:
        print x
        break

    name_idx = name2idx(test_path)
    for x in name_idx.items():
        print x
        break