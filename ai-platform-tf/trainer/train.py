
# python3
# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

import argparse
import json
import hypertune
import os
import warnings

import datetime as datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from pytz import timezone

# from .trainer import model
# from .trainer import inputs

warnings.filterwarnings("ignore")

#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def parse_arguments():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', default=3, type=int, 
                        help='Hyperparameter: depth of network')
    parser.add_argument('--dropout_rate', default=0.02, type=float, 
                        help='Hyperparameter: Drop out rate')
    parser.add_argument('--learning_rate', default=0.0001, type=float, 
                        help='Hyperparameter: initial learning rate')
    parser.add_argument('--batch_size', default=4, type=int, 
                        help='Hyperparameter: batch size of the deep network')
    parser.add_argument('--epochs', default=1, type=int, 
                        help='number of epochs.')
    parser.add_argument('--job-dir', default="",
                        help='Directory to store model checkpoints and logs.')
    parser.add_argument('--train_feature_name', default="",
                        help='GCS path to train feature csv.')
    parser.add_argument('--test_feature_name', default="",
                        help='GCS path to test feature csv.')
    parser.add_argument('--train_label_name', default="",
                        help='GCS path to train label csv.')
    parser.add_argument('--test_label_name', default="",
                        help='GCS path to test label csv.')
    parser.add_argument('--verbosity', choices=['DEBUG','ERROR','FATAL','INFO','WARN'],
                        default='FATAL')
    args, _ = parser.parse_known_args()
    return args

def tf_model(input_dim, output_dim, depth, dropout_rate):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    decr = int((input_dim-output_dim-16)/depth) ^ 1

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    for i in range(1,depth):
        model.add(Dense(input_dim-i*decr, activation=tf.nn.relu, kernel_regularizer='l2'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim, activation=tf.nn.softmax))
    print(model.summary())

    return model

# def custom_loss(y_true, y_pred):
#     custom_loss = mean(square(y_true - y_pred), axis=-1)
#     return custom_loss

# def custom_metric(y_true, y_pred):
#     custom_metric = mean(square(y_true - y_pred), axis=-1)
#     return custom_metric

def get_callbacks(args, early_stop_patience: int = 3):
    """Creates Keras callbacks for model training."""

    # Get trialId
    trialId = json.loads(os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
    if trialId == '':
        trialId = '0'
    print("trialId=", trialId)

    curTime = datetime.datetime.now(timezone('US/Pacific')).strftime('%H%M%S')
    
    # Modify model_dir paths to include trialId
    model_dir = args.job_dir + "/checkpoints/cp-"+curTime+"-"+trialId+"-{val_accuracy:.4f}"
    log_dir   = args.job_dir + "/log_dir"

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    checkpoint_cb  = tf.keras.callbacks.ModelCheckpoint(model_dir, monitor='val_accuracy', mode='max', 
                                                        verbose=0, save_best_only=True,
                                                        save_weights_only=False)
    earlystop_cb   = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    return [checkpoint_cb, tensorboard_cb, earlystop_cb]


if __name__ == "__main__":

    args = parse_arguments()
    print(args)
    print("Input and pre-process data ...")   
    x_train = pd.read_csv(args.train_feature_name)
    y_train = pd.read_csv(args.train_label_name, header=None)
    x_test = pd.read_csv(args.test_feature_name)
    y_test = pd.read_csv(args.test_label_name, header=None)

    print("Shapes:")
    print(x_train.shape)
    print(y_train.shape)
    # Train model
    print("Creating model ...")
    model = tf_model(x_train.shape[1], y_train.shape[1], 
                              depth=args.depth,
                              dropout_rate=args.dropout_rate)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
                     loss='mean_squared_error',
                     metrics=['accuracy'])
    
    print("Fitting model ...")
    callbacks = get_callbacks(args, 3)
    hist = model.fit(np.array(x_train), np.array(y_train), 
                         epochs=args.epochs,
                         batch_size=args.batch_size,
                         validation_data=(np.array(x_test),y_test),
                         callbacks=callbacks)

    # TBD save history for visualization
    final_epoch_accuracy = hist.history['accuracy'][-1]
    final_epoch_count = len(hist.history['accuracy'])

    print('final_epoch_accuracy = %.6f' % final_epoch_accuracy)
    print('final_epoch_count = %2d' % final_epoch_count)

    model.save(args.job_dir)
