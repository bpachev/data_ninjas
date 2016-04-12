from sys import argv, exit
from utils import *
from sklearn.ensemble import ExtraTreesClassifier


train_features, train_labels, test_features, ids, outfile = read_data()

if len(argv) < 5:
 #there are no extra features:
 print "usage trainnpz testnpz extras outfile"
 exit()


n_extras = len(argv) - 4

extra_train = [train_features]
extra_test = [test_features]
for featurefile in argv[3:-1]:
 arch = np.load(featurefile)
 extra_train.append(arch['train'].reshape((len(train_features),1)))
 extra_test.append(arch['test'].reshape((len(test_features),1)))

trainf = np.hstack(extra_train)
testf = np.hstack(extra_test)

np.savez(outfile+"train_bag", features=trainf, labels = train_labels)
np.savez(outfile+"test_bag", features = testf, ids=ids)

print trainf.shape, testf.shape
#extc = ExtraTreesClassifier(n_estimators=1000,max_features= 50,criterion= 'entropy',min_samples_split= 4,
 #max_depth= 35, min_samples_leaf= 2, n_jobs = -1)

#extc.fit(trainf, train_labels)
#probs = extc.predict_proba(testf)[:,1]
#save_submission(outfile, ids=ids, probs = probs)
