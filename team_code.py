#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import sys
import tensorflow as tf
from helper_code import *
from keras.regularizers import l2
from scipy import stats
import keras
import pywt

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 6), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    batch_size = 16                                                                 # make this a number divisible by the total number of samples
    epochs = 10
    units = 12 * batch_size                                                         # number of LSTM cells, hidden states
    input_dim = 1                                                                   # number of features
    num_labels = 2  
    time_step = features.shape[1]
    sample_size = features.shape[0]
    input_shape = (time_step, input_dim) 

    #clear all data from previous runs 
    tf.keras.backend.clear_session()

    #define the model
    model = tf.keras.Sequential([
        #add the layers of the model
         tf.keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True, dropout = 0.2, recurrent_regularizer= l2(0.01)),     # returns a sequence of vectors of dimension batch_size
         tf.keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True, dropout = 0.2, recurrent_regularizer= l2(0.01)),     # returns a sequence of vectors of dimension batch_size
         tf.keras.layers.LSTM(units, dropout = 0.2, recurrent_regularizer= l2(0.01)),                                                     # returns 1xbatch_size
         tf.keras.layers.Dense(num_labels, activation = "softmax")                  # does this need to be units=1 for single output prediction                                                                                            # softmax for multiclass labeling

     ])
    
    inputs = np.random.random((sample_size, time_step, 1))
    output = model(inputs)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) #loss for numerical and multiple matching

    os.makedirs(model_folder, exist_ok=True)

    model.fit(features, labels, epochs = epochs, batch_size = batch_size)
    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.keras')
    model = tf.keras.models.load_model(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    #model = model[model]

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get the model outputs.
    #binary_output = model.predict_step(features)
    #print(binary_output)
    probability_output = model.predict(features)
    binary_output = (probability_output < 0.5).astype(int)
    #print(binary_output)

    

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    # header = load_header(record)
    # age = get_age(header)
    # sex = get_sex(header)
    
    # one_hot_encoding_sex = np.zeros(3, dtype=bool)
    # if sex == 'Female':
    #     one_hot_encoding_sex[0] = 1
    # elif sex == 'Male':
    #     one_hot_encoding_sex[1] = 1
    # else:
    #     one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record) 
    signal = denoise(signal)                
    signal = stats.zscore(signal)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    # num_finite_samples = np.size(np.isfinite(signal))
    # if num_finite_samples > 0:
    #     signal_mean = np.nanmean(signal)
    # else:
    #     signal_mean = 0.0
    # if num_finite_samples > 1:
    #     signal_std = np.nanstd(signal)
    # else:
    #     signal_std = 0.0

    # features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))
    features = np.asarray(signal, dtype=np.float32)

    #return np.asarray(features, dtype=np.float32)
    return features

# Save your trained model.
def save_model(model_folder, model):
    filename = os.path.join(model_folder, 'model.keras')
    model.save(filename)

'''
Denoise the signal data using the wavelet sym4 label, threshold of 0.4,
and found coefficients.

input: list of signal data
return: list of signal data
'''
def denoise(data):
    wavelet_funtion = 'sym3'                                                      #found to be the best function for ECG 

    w = pywt.Wavelet(wavelet_funtion)
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.03                                                               # Threshold for filtering the higher the closer to the wavelet (less noise)

    coeffs = pywt.wavedec(data, wavelet_funtion, level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))

    datarec = pywt.waverec(coeffs, wavelet_funtion)
    return datarec

if __name__ == '__main__':
    train_model('C:\\Users\\yangr\\OneDrive\\Documents\\vscode\\Moody_Challenge\\training_data',
                'C:\\Users\\yangr\\OneDrive\\Documents\\vscode\\Moody_Challenge\\model',
                False)