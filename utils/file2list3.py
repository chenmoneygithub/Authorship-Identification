import os
import re
import random
from unidecode import unidecode

sent_len = 10       # each paragraph should contain at least 5 sentences
para_len = 300      # each author should contain at least 300 paragraphs

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
    regex = re.compile(r"[.?!;][\'\"]?\s")
    for author, newss in auth_newss_dict.items():
        skip = False
        para_list = []
        for news in newss:
            if skip:
                break
            f = open(news, 'r')
            
            for para_idx, paragraph in enumerate(f):
                if para_idx % 3 == 0:
                    word_list_para_level = []
                if skip:
                    break
                paragraph = unidecode(paragraph.decode('utf8')).lower()
                paragraph = paragraph.replace('\n', ' ')
                paragraph = paragraph.replace('--', ' ')
                paragraph = re.sub(pattern=regex, repl=r". ", string=paragraph)
                sentences = paragraph.strip().split(". ")  # split line to sentences
                for sentence in sentences:
                    word_list_sent_level = []
                    words = sentence.strip().split(' ')
                    # print len(words)
                    for word in words:
                        word = word.strip(" .,;:*/+-!?=)(][}{|><@#$&%\"\'0123456789")
                        if word:
                            word_list_sent_level.append(word)
                    if word_list_sent_level:
                        word_list_para_level.append(word_list_sent_level)
                # print len(word_list_para_level)
                if len(word_list_para_level) >= sent_len:
                    temp_list = []
                    temp_list.append(author_idx)
                    temp_list.append(word_list_para_level)
                    # print word_list_para_level
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

if __name__=='__main__':
    cwd = os.getcwd()
    test_path = cwd + '/../dataset/gutenberg'
    res = file2list(test_path)

    print len(res)
    for ii in xrange(len(res)):
        if ii % 100 == 0:
            print res[ii]