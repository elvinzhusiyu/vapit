
import tensorflow as tf
import numpy as np

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

def custom_loss(y_true, y_pred):
    custom_loss = mean(square(y_true - y_pred), axis=-1)
    return custom_loss

def custom_metric(y_true, y_pred):
    custom_metric = mean(square(y_true - y_pred), axis=-1)
    return custom_metric
