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
import keras
import pywt
import scipy

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

    ##Test with small sample
    #num_records = 10
    #features = np.zeros((num_records, 4097,12), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    features = []
    # Iterate over the records.
    
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        #features[i] = extract_features(record)
        features.append(extract_features(record))
        labels[i] = load_label(record)

    sequence = []
    for i in range(num_records):
        #print(features[i].shape)
        sequence.append(features[i].shape[0])

    #print(sequence)
    padded_features = np.zeros((num_records,max(sequence),12),dtype=np.float64)
    for i in range(num_records):
        #print(len(features[i]))
        #padded_features[i] = tf.keras.preprocessing.sequence.pad_sequences(features[i],maxlen=max(sequence))
        #print(features[i].shape[1])
        for signal_index in range(features[i].shape[1]):
           #print(features[i][:,signal_index].shape)
            i_signal = features[i][:,signal_index]
            i_signal=i_signal.reshape(1,i_signal.shape[0])
            padded_features[i][:,signal_index] = tf.keras.preprocessing.sequence.pad_sequences(i_signal,maxlen=max(sequence))
            print(padded_features[i][:,signal_index].shape)
    
    print(padded_features.shape)
    
    # Train the models.
    if verbose:
        print('Training the model on the data...')

    batch_size = 1                                                                 # make this a number divisible by the total number of samples
    epochs = 10
    units = 12 * batch_size                                                         # number of LSTM cells, hidden states
    input_dim = 1                                                                   # number of features
    num_labels = 2  
    length = padded_features.shape[1] #Also known as time_step, this is the length of the signal
    sample_size = padded_features.shape[0]
    concurrent_sig=padded_features.shape[2]
    #sample_size = train_set_datachunk.shape[0]                                      # number of total ECG samples
    #time_step = train_set_datachunk.shape[1]                                        # length of the ECG chunk
    input_shape = (length, concurrent_sig)

    #clear all data from previous runs 
    tf.keras.backend.clear_session()

    #define the model
    model = tf.keras.Sequential([
        #add the layers of the model
         tf.keras.layers.Input(shape=input_shape),
         tf.keras.layers.LSTM(units, return_sequences=True, dropout = 0.2, recurrent_regularizer= l2(0.01)),     # returns a sequence of vectors of dimension batch_size
         tf.keras.layers.LSTM(units, return_sequences=True, dropout = 0.2, recurrent_regularizer= l2(0.01)),     # returns a sequence of vectors of dimension batch_size
         tf.keras.layers.LSTM(units, dropout = 0.2, recurrent_regularizer= l2(0.01)),                            # returns 1xbatch_size
         tf.keras.layers.Dense(num_labels, activation = "softmax")                                                                                                              # softmax for multiclass labeling

     ])
    #inputs = np.random.random((batch_size, time_step, sample_size))
    #output = model(inputs)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) #loss for numerical and multiple matching

    os.makedirs(model_folder, exist_ok=True)

    print("Training")
    model.fit(padded_features, labels, epochs = epochs, batch_size = batch_size)
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
    
    features = features.reshape(-1, len(features), 12)
    print(features.shape)
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
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    
    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    #features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))
    
    #features = np.array(signal[:,1])
    signal = denoise(signal)

    features = np.concatenate((np.full((1,12), age), denoise(signal)))
    #return np.asarray(features, dtype=np.float32)
    return features

# Save your trained model.
def save_model(model_folder, model):
    filename = os.path.join(model_folder, 'model.keras')
    model.save(filename)

def denoise(data):
    wavelet_funtion = 'sym3'                                                      #found to be the best function for ECG 
    data = np.array(data)
    data = data[0::2][:]                                                          #take every other point in the lead signal itself to decrease its length and hopefully fix the memory issue
    shape=data.shape
    #data_cal = len(data[0][0::2])

    datarec = np.zeros((shape[0],shape[1])) 
    for x in range(shape[1]): 
        w = pywt.Wavelet(wavelet_funtion)
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        threshold = 0.03                                                          # Threshold for filtering the higher the closer to the wavelet (less noise)
        coeffs = pywt.wavedec(data[:,x], wavelet_funtion, level=maxlev)
        for i in range(0, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
        sig = pywt.waverec(coeffs, wavelet_funtion)

        if ((shape[0])%2 != 0):                     
            print('This number is odd')                                           #Checks if the number is odd or even
            sig = np.delete(sig, -1)                                              #If odd delete last element


        if shape[0] - len(sig) != 0:       
            print('The shapes are uneven')                                       #Checks to see if the signal and the zeros array are the same size
            if shape[0] - len(sig) > 0:
                sig = np.append(sig, 0)
            if shape[0] - len(sig) < 0:
                sig = np.delete(sig, -1)                                              #If odd delete last element
                

        datarec[:,x] = scipy.stats.zscore(sig)

    return datarec
    

if __name__ == '__main__':
    train_model('C:\\Users\\yangr\\OneDrive\\Documents\\vscode\\Moody_Challenge\\training_data',
                'C:\\Users\\yangr\\OneDrive\\Documents\\vscode\\Moody_Challenge\\model',
                False)