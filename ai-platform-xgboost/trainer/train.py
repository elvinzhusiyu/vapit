

# python3
# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

import argparse
import hypertune
import os
import subprocess
import sys
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier

def train_xgboost(args):
    """ Train a XGBoost model
    Args:
        args: structure with the following field:
            bucket_name, str, gcs bucket name to store trained model
            blob_name, str, gcs blob name to store trained model
            train_feature_name, str, name of the train feature csv
            train_label_name, str, name of train label csv
            no_classes, int, number of prediction classes in the model
            n_estimators, int, number of estimators (hypertune)
            max_depth, int, maximum depth of trees (hypertune)
            booster, str, type of boosters (hypertune)
    Return:
        xgboost model object
    
    """
    
    x_train = pd.read_csv(args.train_feature_name)
    y_train = pd.read_csv(args.train_label_name)
   
    # ---------------------------------------
    # Train model
    # ---------------------------------------

    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'booster': args.booster,
        'min_child_weight': 1,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'reg_alpha': 0,
        'objective': 'multi:softprob',
        'num_class': args.no_classes,
        }
    xgb_model = XGBClassifier(**params, use_label_encoder=False)
    xgb_model.fit(x_train, y_train)

    # ---------------------------------------
    # Save the model to local
    # ---------------------------------------

    temp_name = './model.bst'
    bst = xgb_model.get_booster()
    bst.save_model(temp_name)
    
    # ---------------------------------------
    # Move local model to gcs
    # ---------------------------------------
    
    target_path = os.path.join(args.job_dir, 'model.bst')
    if temp_name != target_path:
        subprocess.check_call(['gsutil', 'cp', temp_name, target_path],
            stderr=sys.stdout)

    return xgb_model
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, help="Required by ai platform training", default='./')
    parser.add_argument("--train_feature_name", type=str, help="Path to training feature csv file")
    parser.add_argument("--train_label_name", type=str, help="Path to training label csv file")
    parser.add_argument("--no_classes", type=int, help="Number of target classes in the label")
    parser.add_argument("--n_estimators", type=int, help="Number of estimators in the xgboost model")
    parser.add_argument("--max_depth", type=int, help="Maximum depth of trees in xgboost")
    parser.add_argument("--booster", type=str, help="Type of booster")
    args = parser.parse_args()
    model = train_xgboost(args)
