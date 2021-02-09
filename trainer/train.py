
# python3
# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

import os
import json
import tensorflow as tf
import numpy as np
import datetime as datetime
from pytz import timezone
import hypertune
import argparse
from trainer import model
from trainer import inputs


import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed


def parse_arguments():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', default=5, type=int, 
                        help='Hyperparameter: depth of net')
    parser.add_argument('--dropout_rate', default=0.2, type=float, 
                        help='Hyperparameter: Drop out rate')
    parser.add_argument('--learning_rate', default=0.00005, type=float, 
                        help='Hyperparameter: initial learning rate')
    parser.add_argument('--batch_size', default=1, type=int, 
                        help='batch size of the deep network')
    parser.add_argument('--epochs', default=2, type=int, 
                        help='epoch.')
    parser.add_argument('--num_samples', default=3000, type=int,
                        help='Number of training samples to use.')
    parser.add_argument('--model_dir', default="",
                        help='Directory to store models and logs.')
    parser.add_argument('--verbosity', choices=['DEBUG','ERROR','FATAL','INFO','WARN'],
                        default='FATAL')
    args, _ = parser.parse_known_args()
    return args


def get_callbacks(args, early_stop_patience: int = 3):
    """Creates Keras callbacks for model training."""

    # Get trialId
    trialId = json.loads(os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
    if trialId == '':
        trialId = '0'
    print("trialId=", trialId)

    curTime = datetime.datetime.now(timezone('US/Pacific')).strftime('%H%M%S')
    
    # Modify model_dir paths to include trialId
    model_dir = args.model_dir + "/checkpoints/cp-"+curTime+"-"+trialId+"-{custom_mse:.4f}"
    log_dir   = args.model_dir + "/log_dir"

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    checkpoint_cb  = tf.keras.callbacks.ModelCheckpoint(model_dir, monitor='custom_mse', mode='min', 
                                                        verbose=0, save_best_only=True,
                                                        save_weights_only=False)
    earlystop_cb   = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    return [checkpoint_cb, tensorboard_cb, earlystop_cb]


if __name__ == "__main__":

    # ---------------------------------------
    # Parse Arguments
    # ---------------------------------------
    args = parse_arguments()
    #args.model_dir = MODEL_DIR + datetime.datetime.now(timezone('US/Pacific')).strftime('/model_%m%d%Y_%H%M')
    print(args)

    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

    # ---------------------------------------
    # Input Data & Preprocessing
    # ---------------------------------------
    print("Input and pre-process data ...")
    # Extract train_seismic, train_label
    data_dir = 'gs://../images'
    label_dir = 'gs://../labels'
    train_dataset = inputs.load_data(data_dir, label_dir, range(0,args.num_samples), args.batch_size)
    
    # ---------------------------------------
    # Train model
    # ---------------------------------------
    print("Creating model ...")
    tf_model = model.tf_model(depth=args.depth,
                              dropout_rate=args.dropout_rate)
    tf_model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
                     loss=model.custom_loss,  # loss is the custom loss I shared with you
                     metrics=[model.custom_mse])
    
    print("Fitting model ...")
    callbacks = get_callbacks(args, 3)
    history = tf_model.fit(train_dataset, 
                           epochs=args.epochs,
                           validation_split = 0.0,
                           callbacks=callbacks)

    # TBD save history for visualization

    final_epoch_accuracy = history.history['custom_mse'][-1]
    final_epoch_count = len(history.history['custom_mse'])

    print('final_epoch_accuracy = %.6f' % final_epoch_accuracy)
    print('final_epoch_count = %02d' % final_epoch_count)
