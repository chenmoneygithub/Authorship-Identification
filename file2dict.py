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
                word = word.strip(" ,\"\'")         # strip a word
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
            news_vec_list.append(news_vec)
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
            news_vec_list.append(news_vec)
        portfolio[idx] = news_vec_list
        idx += 1
    return portfolio



"""
following is used for test
"""
if __name__=='__main__':
    from timeit import Timer
    t1=Timer("extract_name(\"./dataset/C50/C50train\")","from __main__ import extract_name")
    t2=Timer("extract_num(\"./dataset/C50/C50train\")","from __main__ import extract_num")
    print t1.timeit(1)
    print t2.timeit(1)

    cwd = os.getcwd()
    test_path = cwd + '/dataset/C50/C50train'
    output_name = extract_name(test_path)
    for x in output_name.items():
        print x
        break
    output_num = extract_num(test_path)
    for x in output_num.items():
        print x
        break