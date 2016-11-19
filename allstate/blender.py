import argparse
import utils
import numpy as np
from sys import exit
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple datasets to make one submission or a new feature")
    parser.add_argument("dataset_files", type=argparse.FileType('r'), nargs="+")
    parser.add_argument("--output_type", type=str, default="submission")
    parser.add_argument("outfile", type=argparse.FileType('w'))
#    parser.add_argument("")
    args = parser.parse_args()
    
    is_submission = False
    if args.output_type == "dataset":
        is_submission = False
    elif args.output_type == "submission":
        is_submission = True
    else:
        raise ValueError("Must specify output type to be submission or dataset.")
    
    trainf, trainl, testf, test_ids, feature_names = utils.combine_datasets(args.dataset_files)
    

    shift = 200
    y = np.log(trainl + shift)
    ids = test_ids
    
    RANDOM_STATE = 2016
    params = {
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE
    }

#    xgtrain = xgb.DMatrix(trainf, label=y)
#    xgtest = xgb.DMatrix(testf)

#    res = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5, stratified=False,
#         early_stopping_rounds=50, verbose_eval=1, show_stdv=True, feval=evalerror, maximize=False)
    params["num_rounds"] = int(2 / 0.9)
    params["feval"] = evalerror
    pred, trainpred = utils.cv_xgboost(params, trainf, y, testf)
    pred = np.exp(pred) - shift
    train_pred = np.exp(trainpred) - shift
    utils.save_submission("data/blended1.csv", ids=ids, loss=pred)
    if is_submission: utils.save_submission(args.outfile, ids=ids, loss=pred)
    else:
        pred = pred.reshape((len(pred),1))
        train_pred = train_pred.reshape((len(train_pred),1))
        save_dataset(args.outfile, train_features=train_pred, train_labels=trainl, test_features=pred, ids=ids, feature_names=['xgb'])
    
