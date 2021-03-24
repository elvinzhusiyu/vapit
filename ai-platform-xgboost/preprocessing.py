
# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================
# Author: Elvin Zhu
# Email: elvinzhu@google.com

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import logging

BUCKET_NAME = 'tuti_asset'
BLOB_NAME = 'datasets/mortgage_structured.csv'
TARGET_COLUMN = 'TARGET'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_preprocess():
    input_file = 'gs://{}/{}'.format(
        BUCKET_NAME, 
        BLOB_NAME, 
        )
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
    n_classes = dataset[TARGET_COLUMN].nunique()
    logging.info("No. of Classes: {}".format(n_classes))

    # Split with a small test size so as to allow our model to train on more data
    logging.info("Perform train/test split ...")
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.drop(TARGET_COLUMN, axis=1), 
        dataset[TARGET_COLUMN], 
        test_size=0.1,
        random_state=1,
        shuffle=True, 
        stratify=dataset[TARGET_COLUMN], 
        )

    logging.info("Get feature/label shapes ...")
    logging.info("x_train shape = {}".format(x_train.shape))
    logging.info("x_test shape = {}".format(x_test.shape))
    logging.info("y_train shape = {}".format(y_train.shape))
    logging.info("y_test shape = {}".format(y_test.shape))

    base_name, ext_name = os.path.splitext(input_file)
    x_train_name = "{}_x_train{}".format(base_name, ext_name)
    x_test_name = "{}_x_test{}".format(base_name, ext_name)
    y_train_name = "{}_y_train{}".format(base_name, ext_name)
    y_test_name = "{}_y_test{}".format(base_name, ext_name)

    x_train.to_csv(x_train_name, index=False)
    x_test.to_csv(x_test_name, index=False)
    y_train.to_csv(y_train_name, index=False)
    y_test.to_csv(y_test_name, index=False)
    
    # Saving data
    logging.info("Saving data ...")
    logging.info("x_train saved to {}".format(x_train_name))
    logging.info("x_test saved to {}".format(x_test_name))
    logging.info("y_train saved to {}".format(y_train_name))
    logging.info("y_test saved to {}".format(y_test_name))
    logging.info("finished")
    return n_classes

if __name__ == "__main__":
    data_preprocess()
