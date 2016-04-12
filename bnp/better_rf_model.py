from utils import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, StratifiedKFold


train_features, train_labels, test_features, ids, outfile = read_data()

model = RandomForestClassifier(n_estimators=1000, criterion= "entropy",n_jobs=-1, max_depth = 35, min_samples_split=4, min_samples_leaf = 2)
for i in xrange(5):
    probs, tprobs = cross_val_model(model, train_features, train_labels, test_features)
    np.savez(outfile+"_feature"+str(i), train=tprobs, test=probs)
save_submission(outfile, ids=ids, probs=probs)
print "saved model, validating now"
print cross_val_score(model, train_features, train_labels, scoring= "log_loss")
