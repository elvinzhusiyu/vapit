
# python3
# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from google.cloud import storage
from datetime import datetime
from pytz import timezone
import hypertune
from sklearn import metrics
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    INPUT_FILE_NAME = 'Step10_Final_dataset.csv'
    BUCKET_NAME = 'tuti_asset'
    FOLDER_NAME = 'datasets'
    _TARGET_COLUMN = 'TARGET'

    input_file = 'gs://' + BUCKET_NAME + '/' + FOLDER_NAME + '/' + INPUT_FILE_NAME

    # Read the data
    try:
        dataset = pd.read_csv(input_file)
    except:
        print("Oops! That is invalid filename. Try again...")

    print(dataset.shape)

    # ---------------------------------------
    # Pre-processing code from customer
    # ---------------------------------------

    # Drop useless columns
    dataset.drop(['LOAN_SEQUENCE_NUMBER'], axis=1, inplace=True)

    # Inputs to an XGBoost model must be numeric. One hot encoding was previously found to yield better results 
    # than label encoding for the particular
    strcols = [col for col in dataset.columns if dataset[col].dtype == 'object']
    dataset = pd.get_dummies(dataset, columns=strcols)

    # Train Test Split and write out the train-test files

    # Split with a small test size so as to allow our model to train on more data
    X_train, X_test, y_train, y_test = \
        train_test_split(dataset.drop(_TARGET_COLUMN, axis=1), 
                                      dataset[_TARGET_COLUMN], stratify=dataset[_TARGET_COLUMN], 
                                      shuffle=True, test_size=0.2
                                     )
    print("X_train shape = ", X_train.shape)
    print("X_test  shape = ", X_test.shape)

    # count number of classes
    values, counts = np.unique(y_train, return_counts=True)
    NUM_CLASSES = len(values)

    # ---------------------------------------
    # Train model
    # ---------------------------------------

    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'booster': 'gbtree',
        'min_child_weight': 1,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'reg_alpha': 0,
        'objective': 'multi:softprob',
        'num_class': NUM_CLASSES
        }
    xgb_model = XGBClassifier(**params)
    #xgb_model.set_params(**params)
    xgb_model.fit(X_train, y_train)

    # ---------------------------------------
    # Save the model to GCS
    # ---------------------------------------

    bst_filename = 'model.bst'
    bst = xgb_model.get_booster()
    bst.save_model(bst_filename)
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob('{}/{}'.format(
        datetime.now().strftime(FOLDER_NAME+'/models/model_%Y%m%d_%H%M%S'),
        bst_filename))
    blob.upload_from_filename(bst_filename)

    # predict the model with test file
    y_pred = xgb_model.predict(X_test)

    # Binarize multiclass labels
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    # Define the score we want to use to evaluate the classifier on
    #score = metrics.accuracy_score(y_test, y_pred)
    #score = metrics.average_precision_score(y_test, y_pred, average='macro')
    #score = metrics.f1_score(y_test, y_pred, average='macro')
    #score = metrics.fbeta_score(y_test, y_pred, average='macro', beta=0.5)
    #score = metrics.hamming_loss(y_test, y_pred)
    #score = metrics.log_loss(y_test, y_pred)
    #score = metrics.precision_score(y_test, y_pred, average='macro')
    #score = metrics.recall_score(y_test, y_pred, average='macro')
    score = metrics.roc_auc_score(y_test, y_pred, average='macro')
    #score = metrics.zero_one_loss(y_test, y_pred)

    # The default name of the metric is training/hptuning/metric. 
    # We recommend that you assign a custom name. The only functional difference is that 
    # if you use a custom name, you must set the hyperparameterMetricTag value in the 
    # HyperparameterSpec object in your job request to match your chosen name.
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='roc_auc',
        metric_value=score,
        global_step=1000
    )
