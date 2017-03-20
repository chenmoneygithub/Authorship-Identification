import re
import numpy as np


# /Users/kevinhe/Documents/Stanford\ Academic/2016-17\ Winter/CS224n\ Natural\ Language\ Processing\ with\ Deep\ Learning/Project\ path/Authorship-Identification


class TrainingHistory:
    def __init__(self,l,numbers):
        print l
        self.model=l[0]
        self.starting_time=l[1]
        self.cell_type=l[2]
        self.embed_size=l[3]
        self.hidden_size=l[4]
        self.learning_rate=l[5]
        self.regularization=l[6]
        self.batch_size=l[7]
        numbers_np=np.array(numbers)
        self.epoch=(numbers_np[:,0].astype(int))+1
        self.loss=numbers_np[:,1]
        self.train_accu=numbers_np[:,2]
        self.test_accu=numbers_np[:,3]

def convert_line_to_list(a_line):
    l=a_line.split(" ")
    l=[re.sub(r'\t', '', s) for s in l if s!='']
    l=[re.sub(r'\n', '', s) for s in l if s!='']
    return l

def parse_file(f):
    lines=f.readlines()
    new_list=[]
    all_th=[]
    numbers=[]
    for a_line in lines:
        l=convert_line_to_list(a_line)
        # print l
        if l==['END']:
            all_th.append(TrainingHistory(new_list,numbers))
        elif l==['']:
            continue
        elif l[0]=="###":
            new_list=[]
        elif l[0]=="training_model:":
            new_list.append(l[1])
        elif l[0]=="starting_time:":
            new_list.append(l[1]+' '+l[2])
        elif l[0]=="cell_type:":
            new_list.append(l[1])
        elif l[0]=="embed_size:":
            new_list.append(int(l[1]))
        elif l[0]=="hidden_size:":
            new_list.append(int(l[1]))
        elif l[0]=="learning_rate:":
            new_list.append(float(l[1]))
        elif l[0]=="regularization:":
            new_list.append(float(l[1]))
        elif l[0]=="batch_size:":
            new_list.append(int(l[1]))
        elif len(l)==4 and l[0]=="epoch":
            numbers=[]
        elif len(l)==4 and l[0]!='starting':
            numbers.append([float(num) for num in l])
    return all_th
