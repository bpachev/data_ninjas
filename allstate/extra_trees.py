from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import utils
trainf, trainl, testf, test_ids, feature_names = utils.read_dataset('data/no_big_cats1.npz')
extra_trees = ExtraTreesRegressor(n_estimators = 100, max_features= .3,min_samples_split= 8,
 max_depth = 40, min_samples_leaf= 2, n_jobs = 4)
y, trainy = utils.cross_val_model(extra_trees, trainf, trainl, testf)
y = y.reshape((len(y),1))
ytrain = ytrain.reshape((len(ytrain), 1))
utils.save_dataset("data/extra_trees1.npz",ytrain, trainl, y, test_ids, ['extra_trees'])
