import os
import random
from unidecode import unidecode

word_len = 40
para_len = 300


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
        elif author == 'README.txt':
            pass
        else:
            newss = os.listdir(input_dir + '/' + author)
            news_list = []
            for news in newss:                  # traverse each txt file in sub-dir
                if news[0] == '.':
                    pass
                else:
                    news_dir = input_dir + '/' + author + '/' + news
                    news_list.append(news_dir)
            author_newss_dict[author] = news_list
    return author_newss_dict


def file2list(input_dir):
    auth_para_list = []
    auth_newss_dict = read_mydir(input_dir)
    author_idx = 0
    for author, newss in auth_newss_dict.items():
        skip = False
        para_list = []
        for news in newss:
            # print newss             # here is OK
            if skip:
                break
            f = open(news, 'r')

            for paragraph in f:
                word_list = []
                if skip:
                    break
                paragraph = unidecode(paragraph.decode('utf8')).lower()
                paragraph = paragraph.replace('\n', ' ')
                paragraph = paragraph.replace('--', ' ')
                words = paragraph.strip().split(' ')
                for word in words:
                    word = word.strip(" .,;:*/+-!?=)(][}{|><@#$&%\"\'0123456789")
                    if word:
                        word_list.append(word)

                if len(word_list) >= word_len:
                    temp_list = []
                    temp_list.append(author_idx)
                    temp_list.append(word_list)
                    para_list.append(temp_list)             # for this author
                    # auth_para_list.append(temp_list)        # for all authors
                if len(para_list) >= para_len:
                    random.shuffle(para_list)
                    auth_para_list.extend(para_list)
                    skip = True
        author_idx += 1
        if not skip:
            print "WARNING: this author --", author, "does not meet the paragraph requirement"
    return auth_para_list

cwd = os.getcwd()
test_path = cwd + '/../dataset/gutenberg'
res = file2list(test_path)

print len(res)
for ii in xrange(len(res)):
    if ii % 100 == 0:
        print res[ii]