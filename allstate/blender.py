import argparse
import utils
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple datasets to make one submission or a new feature")
    parser.add_argument("dataset_files", type=argparse.FileType('r'), nargs="+")
    parser.add_argument("--output_type", type=str, default="submission")
    parser.add_argument("outfile", type=argparse.FileType('w'))
    args = parser.parse_args()
    
    is_submission = False
    if args.output_type == "dataset":
        is_submission = False
    elif args.output_type == "submission":
        is_submission = True
    else:
        raise ValueError("Must specifiy output type to be submission or dataset.")
    
    trainf_list = []
    testf_list = []
    trainl = None
    test_ids = None
    feature_name_list = []
    for f in args.dataset_files:
        trainf, trainl, testf, test_ids, feature_names = utils.read_dataset(f)
        trainf_list.append(trainf)
        testf_list.append(testf)
        feature_name_list.append(feature_names)
    
    trainf = np.hstack(trainf_list)
    testf = np.hstack(testf_list)
    feature_names = np.hstack(feature_name_list)    
    print feature_names
    print testf.shape
    print trainf.shape
