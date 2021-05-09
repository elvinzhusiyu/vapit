
# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================
# Author: Elvin Zhu
# Email: elvinzhu@google.com

import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# input_file = "gs://tuti_asset/datasets/mortgage_structured.csv"
# x_train_name = "gs://tuti_asset/datasets/mortgage_structured_x_train.csv"
# x_test_name = "gs://tuti_asset/datasets/mortgage_structured_x_test.csv"
# y_train_name = "gs://tuti_asset/datasets/mortgage_structured_y_train.csv"
# y_test_name = "gs://tuti_asset/datasets/mortgage_structured_y_test.csv"
# target_column = 'TARGET'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_preprocess(
    input_file, 
    x_train_name, 
    y_train_name, 
    x_test_name, 
    y_test_name,
    target_column):
    """ Preprocess input data by
    1. Dropping unique ID column;
    2. Convert categorical into one-hot encodings;
    3. Count number of unique classes;
    4. Split train/test
    5. Remove na with zeros
    6. Save process data into gcs
    
    Args:
        input_file: string, input file gcs path
        x_train_name: str, feature csv for training,
        y_train_name: str, label csv for training,
        x_test_name: str, feature csv for testing,
        y_test_name: str, label csv for testing
        target_column: str, columns name which contains training labels
    Return:
        No. of classes
    """

    logging.info("Preprocessing raw data:")
    logging.info(" => Drop id column:")
    logging.info(" => One hot encoding categorical features")
    logging.info(" => Count number of classes")
    logging.info(" => Perform train/test split")

    logging.info("Reading raw data file: {}".format(input_file))
    dataset = pd.read_csv(input_file)
    # Drop unique id column which is not useful for ML
    logging.info("Drop unique id column which is not an useful feature for ML: {}".format('LOAN_SEQUENCE_NUMBER'))
    dataset.drop(['LOAN_SEQUENCE_NUMBER'], axis=1, inplace=True)

    # Convert categorical columns into one-hot encodings
    logging.info("Convert categorical columns into one-hot encodings")
    [logging.info("categorical feature: {}".format(col)) for col in dataset.columns if dataset[col].dtype == 'object']
    str_cols = [col for col in dataset.columns if dataset[col].dtype == 'object']
    dataset = pd.get_dummies(dataset, columns=str_cols)
    
    # Count number of unique classes
    logging.info("Count number of unique classes ...")
    n_classes = dataset[target_column].nunique()
    logging.info("No. of Classes: {}".format(n_classes))

    # Split with a small test size so as to allow our model to train on more data
    logging.info("Perform train/test split ...")
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.drop(target_column, axis=1), 
        dataset[target_column], 
        test_size=0.1,
        random_state=1,
        shuffle=True, 
        stratify=dataset[target_column], 
        )

    # Fill Nan value with zeros
    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)
    
    logging.info("Get feature/label shapes ...")
    logging.info("x_train shape = {}".format(x_train.shape))
    logging.info("x_test shape = {}".format(x_test.shape))
    logging.info("y_train shape = {}".format(y_train.shape))
    logging.info("y_test shape = {}".format(y_test.shape))

    x_train.to_csv(x_train_name, index=False)
    x_test.to_csv(x_test_name, index=False)

    # The preprocessing for label column is different 
    # between tensorflow and XGBoost models
    pd.get_dummies(y_train).to_csv(y_train_name, index=False, header=None)
    pd.get_dummies(y_test).to_csv(y_test_name, index=False, header=None)
    
    # Saving data
    logging.info("Saving data ...")
    logging.info("x_train saved to {}".format(x_train_name))
    logging.info("y_train saved to {}".format(y_train_name))
    logging.info("x_test saved to {}".format(x_test_name))
    logging.info("y_test saved to {}".format(y_test_name))
    logging.info("finished")
    return n_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse arguments for preprocessing')
    parser.add_argument('--input_file', type=str, help="Input file gcs path")
    parser.add_argument('--x_train_name', type=str, help="feature csv for training")
    parser.add_argument('--x_test_name', type=str, help="feature csv for testing")
    parser.add_argument('--y_train_name', type=str, help="label csv for training")
    parser.add_argument('--y_test_name', type=str, help="label csv for testing")
    parser.add_argument('--target_column', type=str, help="columns name which contains training labels")    
    args = parser.parse_args()
    
    data_preprocess(
        args.input_file,
        args.x_train_name,
        args.y_train_name,
        args.x_test_name,
        args.y_test_name,
        args.target_column,
    )
