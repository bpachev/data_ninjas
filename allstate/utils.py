from sys import argv, exit
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.cross_validation import StratifiedKFold

def cross_val_model(model, train_features, labels, test_features, nfolds = 5, nbags = 1):
    """
    model -- the model to train
    train_features -- the training features
    labels -- the variable to predict
    test_features -- the testing features
    nfolds -- number of cross-val folds to use
    nbags -- the number of times to randomly repeat and average the result
    NOTE: this is for regression problems with mean average error loss
    """
    y = np.zeros(len(test_features))
    trainy = np.zeros(len(train_features))
    for bag in xrange(nbags):
        i=0
        skf = StratifiedKFold(labels, n_folds=nfolds, random_state=int(time()))
        for train_mask, test_mask in skf:
            i+=1
            model.fit(train_features[train_mask], labels[train_mask])
            y += model.predict(test_features)[:,1]
            trainy[test_mask] = model.predict(train_features[test_mask])[:,1]
            print "Finished cross val fold %d with validation error %f, bag %d" % (i, mean_absolute_error(trainy[labels_test], mask[test_mask] ), d)
    y /= (nbags*nfolds)
    trainy /= nbags
    return y, trainy

def extract_mat(df, cat, cont):
    """
    Create a numpy matrix consisting a subset of the columns of a pandas dataframe
    cat is the categorical column names
    cont is the continuous column names
    """
    arrs = []
    for col in cont:
        arrs.append(np.array(df[col]))    
    for col in cat:
        arrs.append(pd.get_dummies(df[col]).as_matrix().T)
    return np.vstack(arrs).T    


def save_dataset(filename, train_labels, train_features, test_features, ids, feature_names):
    """
    ids  are the ids for the test dataset
    filename is the npz datafile to write to
    """
    np.savez(filename, train_labels=train_labels, train_features=train_features, test_features=test_features, ids=ids, feature_names=feature_names)

def read_dataset(dataset):
    """
    dataset -- the numpy archive with the training and test dataset
    """
    arch = np.load(dataset)
    train_features = arch['train_features']
    train_labels = arch['train_labels']
    test_features = arch['test_features']    
    ids = arch['ids']
    feature_names = arch['feature_names'] 
    return train_features, train_labels, test_features, ids, feature_names

def save_submission(filename, ids=None, loss=None):
    pd.DataFrame({"id":ids, "loss":loss}).to_csv(filename, index=False)
