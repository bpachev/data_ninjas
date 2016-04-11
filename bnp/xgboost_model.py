import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) < 4:
    print "Usage: training file test file output file"
    exit()

def xgb_params():
    params = {}
    params['eta'] = .2
    params['objective'] = 'binary:logistic'
    params['silent'] = 1
    params["eval_metric"] = 'logloss'
#    params['alpha'] = 10
    params['max_depth'] = 5
    params['lambda'] = 2
    params['colsample_bytree'] = .3
    params['subsample'] = 1.
    return params
#    params[]
num_rounds = 100
def train_model(features, labels, validation=None):
 xgmat = xgb.DMatrix(features, label=labels)
 #create params
 params = xgb_params()
 wlist = [(xgmat, 'train')]
 if validation is not None:
     wlist.append((validation, 'validation'))
 bst = xgb.train(params, xgmat, num_rounds, wlist)
 return bst

def  validate_model(features, labels):
 xgmat = xgb.DMatrix(features, label=labels)
 #create params
 params = xgb_params()

 print xgb.cv(params, xgmat, num_rounds, metrics=["logloss"])


outfile = argv[3]
trainfile = argv[1]
testfile = argv[2]

test_arch = np.load(testfile)
train_arch = np.load(trainfile)
train_features = train_arch['features']
train_labels = train_arch['labels']
test_features = test_arch['features']
ids = test_arch['ids']
print "loaded data"
xgbtest = xgb.DMatrix(test_features)

def train_and_save_folds(outfile, nfolds = 3, random_state = 1):
 fold = StratifiedKFold(train_labels, n_folds=nfolds, random_state = random_state)
 probs = np.zeros(len(ids))
 for train_inds, test_inds in fold:
     validation  = xgb.DMatrix(train_features[test_inds], label = train_labels[test_inds])
     model = train_model(train_features[train_inds], train_labels[train_inds], validation=validation)
     probs += model.predict(xgbtest)
 probs /= nfolds
 save(outfile, probs)

def save(outfile, probs):
    res = np.zeros((len(ids), 2))
    res[:,0]  = ids
    res[:,1] = probs
    np.savetxt(outfile, res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")

def train_and_save(outfile):
 model = train_model(train_features, train_labels)
 #assert len(ids) == len(test_features)
# model.save_model("xgboost_bnp4.model")
 print "Trained"
 save(outfile, model.predict(xgbtest))

#for i in xrange(50):
 # train_and_save_folds(outfile+"_"+str(i)+".csv", random_state = i)
print "validating"
validate_model(train_features, train_labels)

