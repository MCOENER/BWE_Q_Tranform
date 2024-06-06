# import pandas as pd
import numpy as np
# import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
#from tensorflow.keras.layers import Dense, InputLayer, Flatten
#from tensorflow.keras.models import Sequential, Model
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
#from pathlib import Path
#import glob
#from glob import glob
import librosa
#from IPython.display import Audio
from pystoi import stoi
import random
from scipy.signal import butter, filtfilt
from keras.models import load_model
#from sklearn import decomposition
import soundfile as sf

#########################################################################
def gpu_usage() :
    np.random.seed(1337)  # for reproducibility
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_physical_devices('GPU')
            print(len(gpus), 'Physical GPUs', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)
gpu_usage()
#########################################################################
def load_dataset(target_folder, dur, fs):
    res = []
    for dir in os.listdir(target_folder):
        for file in os.listdir(os.path.join(target_folder, dir)):
            if file.endswith(".wav"):
                res.append(os.path.join(target_folder, dir, file))



    np_arr = []

    for file in res:
        y, _ = librosa.load(file, sr=fs, duration=dur)
        samples = dur * fs
        size = len(y)
        if samples > size:
            y = np.append(y, np.zeros(samples - size))
        elif samples < size:
            y = y[:samples]
        np_arr.append(y)
    return np_arr
#########################################################################
def aud_to_cqt(np_arr, sr, n_bins):
    cqt_mat = []
    for i in np_arr:
        cqt = np.abs(librosa.cqt(i, sr=sr, n_bins=n_bins))
        cqt_mat.append(cqt)
    return cqt_mat
#########################################################################
def icqt(C, sr, hop_length, fmin):
    return librosa.icqt(C, sr=sr, hop_length=hop_length, fmin=fmin)
#########################################################################
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
#########################################################################
def high_to_low(np_arr, cutoff, fs, order):
    lowpass_nparr = []
    for i in np_arr:
        arr = butter_lowpass_filter(i, cutoff, fs, order)
        lowpass_nparr.append(arr)
    return lowpass_nparr
#########################################################################
def comp_stoi(clean, predicted, fs):
    if len(clean) >= len(predicted):
        return stoi(clean[:len(predicted)], predicted, fs, extended=False)
    return stoi(clean, predicted[:len(clean)], fs, extended=False)
#########################################################################
def equalise(orig, proc):
    if len(orig) >= len(proc):
        return orig[:len(proc)], proc
    return orig, proc[:len(orig)]
#########################################################################
def test_train_split_cqt(target_folder, ratio, dur, fs, cutoff, order, no_cqt, highaud, lowaud):
    highcqt = aud_to_cqt(highaud, fs, no_cqt)
    lowcqt = aud_to_cqt(lowaud, fs, no_cqt)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_size = int(ratio * len(lowaud))
    for i in lowcqt[:train_size]:
        for j in range(len(i[0])):
            arr = []
            for z in range(len(i)):
                arr.append(i[z][j])
            x_train.append(arr)

    train_len = len(x_train)
    x_train = np.array(x_train).reshape(train_len, no_cqt)

    x_test = []
    for i in lowcqt[train_size:]:
        for j in range(len(i[0])):
            arr = []
            for z in range(len(i)):
                arr.append(i[z][j])
            x_test.append(arr)

    test_len = len(x_test)
    x_test = np.array(x_test).reshape(test_len, no_cqt)

    y_train = []
    for i in highcqt[:train_size]:
        for j in range(len(i[0])):
            arr = []
            for z in range(len(i)):
                arr.append(i[z][j])
            y_train.append(arr)

    train_len = len(y_train)
    y_train = np.array(y_train).reshape(train_len, no_cqt)

    y_test = []
    for i in highcqt[train_size:]:
        for j in range(len(i[0])):
            arr = []
            for z in range(len(i)):
                arr.append(i[z][j])
            y_test.append(arr)

    test_len = len(y_test)

    y_test = np.array(y_test).reshape(test_len, no_cqt)

    return x_train, y_train, x_test, y_test
#########################################################################


#########################################################################
def pred_cqt(test_aud, lowaud, model_path, fs, no_cqt):
    model = load_model(model_path)
    cqt_mat = np.abs(librosa.cqt(lowaud[test_aud], sr=fs, n_bins=no_cqt))
    lowcqt = []
    for i in range(len(cqt_mat[0])):
        arr = []
        for j in range(len(cqt_mat)):
            arr.append(cqt_mat[j][i])
        lowcqt.append(arr)
    highcqt = model.predict(np.array(lowcqt))
    highcqt = np.transpose(highcqt)
    final_aud = icqt(highcqt, sr=fs, hop_length=512, fmin=librosa.note_to_hz('C1'))

    return final_aud

#########################################################################

#########################################################################
target_folder = r"E:\Projects_2024\adjusted_data_mert"

fs = 16000
cutoff = 4100
nyq = 0.5 * fs
order = 6
dur = 4

no_cqt = 84  # This is an example, adjust based on your needs
input_shape = (no_cqt, 1)

ratio = 0.7
num_classes = no_cqt

highaud = load_dataset(target_folder, dur, fs)
lowaud = high_to_low(highaud, cutoff, fs, order)
x_train, y_train, x_test, y_test = test_train_split_cqt(target_folder, ratio, dur, fs, cutoff, order, no_cqt, highaud, lowaud)

print(f"This is x_train: {x_train}")
################################################################################################################################
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(900, activation="relu"),
        layers.Dense(800, activation="relu"),
        layers.Dense(600, activation="relu"),
        layers.Dense(400, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(400, activation="relu"),
        layers.Dense(600, activation="relu"),
        layers.Dense(800, activation="relu"),
        layers.Dense(900, activation="relu"),
        layers.Dense(num_classes, activation="linear"),
    ]
)
model.summary()
################################################################################################################################
batch_size = 36
epochs = 1
test_aud = 300

loss_func = tf.keras.losses.CosineSimilarity(axis=1)

loss_func = tf.keras.losses.MeanSquaredError()
adam = tf.keras.optimizers.Adam()
model.compile(loss=loss_func, optimizer=adam, metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
################################################################################################################################
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
################################################################################################################################
model.save('curr_model_cosine.h5')
model_path = 'curr_model_cosine.h5'
################################################################################################################################
ergebnis_dir = "F:\Mert\ergebnis"
os.makedirs(ergebnis_dir, exist_ok=True)

# Save all files in the dataset and save each predicted file
for test_aud in range(len(lowaud)):
    # Save original test audio file
    test_aud_file = highaud[test_aud]
    output_file = os.path.join(ergebnis_dir, f"test_aud_{test_aud}.wav")
    sf.write(output_file, test_aud_file, fs)
#    print(f"Die Datei {output_file} wurde gespeichert.")

    # Save lowpass filtered audio file
    low_aud_file = lowaud[test_aud]
    output_file = os.path.join(ergebnis_dir, f"low_aud_{test_aud}.wav")
    sf.write(output_file, low_aud_file, fs)
#    print(f"Die Datei {output_file} wurde gespeichert.")

    # Save high-quality original audio file
    high_aud_file = highaud[test_aud]
    output_file = os.path.join(ergebnis_dir, f"high_aud_{test_aud}.wav")
    sf.write(output_file, high_aud_file, fs)
#    print(f"Die Datei {output_file} wurde gespeichert.")

    # Save model prediction
    final_aud = pred_cqt(test_aud, lowaud, model_path, fs, no_cqt)
    output_file = os.path.join(ergebnis_dir, f"final_aud_{test_aud}.wav")
    sf.write(output_file, final_aud, fs)
#    print(f"Die Datei {output_file} wurde gespeichert.")

# Compute and print the STOI value for the last audio file
# print("STOI:" + comp_stoi(highaud[test_aud], final_aud, fs))