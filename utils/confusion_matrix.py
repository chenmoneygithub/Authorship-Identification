import random
from util import ConfusionMatrix

"""
input: real, a list of example class, [0, 3, 5, 2, 8, 10 ....]
       pred, a list of predict class, [0, 3, 4, 2, 8, 12 ....]
output: confusion matrix
"""

def generate_cm(real, pred, n_class):
    LBLS = [str(x) for x in xrange(n_class)]
    token_cm = ConfusionMatrix(labels=LBLS)
    for l, l_ in zip(real, pred):
        token_cm.update(l, l_)          # self.counts[gold][guess] += 1
    return token_cm


# for test
# real = [s for s in xrange(50)] * 10
# real.extend([random.randint(0, 49) for r in xrange(1000)])
# pred = [p for p in xrange(50)] * 10
# pred.extend([random.randint(0,49) for r in xrange(1000)])
#
# t_cm = generate_cm(real, pred, 50)
# print t_cm.as_table()
# print t_cm.summary()
# print t_cm.as_matrix()