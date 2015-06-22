#!/usr/bin/python
#-*- coding:utf-8 -*-

import pandas
import numpy
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier


DATAPATH = "data_logit.csv"
USE_COL_END = 2


    
df = pandas.read_csv(DATAPATH,
                     header=None,
                         names=range(6))
    
target = df.ix[:, 0]
data = df.ix[:, :1]
    
print "end load"
for C in [5**i for i in xrange(-3, 4)]:
    """
        model = SGDClassifier(alpha=C,
        loss="log",
        n_iter=50,
        n_jobs=-1,
        penalty='l1',
        #penalty='elasticnet',
        class_weight='auto',
        random_state=0)
    """

    model = LogisticRegression(C=C,
                                   penalty='l1',
								   class_weight='auto'
                                   )
        
    scores = cross_val_score(model,
                                 data,
                                 target,
                                 n_jobs=-1,
                             scoring='roc_auc'
                                 )
    print "C:%s auc:%s"%(C, scores.mean())

