import numpy as np
import utils.data_util as data_util
import matplotlib.pyplot as plt

print 'this is numpy playground'


confusion=np.array([
[15,1,2],
[2,12,6],
[7,1,20]])

allF1=data_util.calculate_F1_all(confusion)
print allF1
print data_util.calculate_F1(confusion,'macro')

data_util.visualize_cm(confusion)
