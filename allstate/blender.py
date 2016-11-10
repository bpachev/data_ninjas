import argparse
import utils
import numpy as np
from sys import exit

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
        raise ValueError("Must specifiy output type to be submission or dataset.")
    
       
    trainf, trainl, testf, test_ids, feature_names = utils.combine_datasets(args.dataset_files)
    
