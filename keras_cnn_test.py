import tensorflow as tf
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
import cv2
import random
import keras
sys.path.append('auto_gen_from_tf')
import deepletters_keras as kit_model

mean_image = np.resize(np.load('CNN/mean_image.npy'), [227,227,3])

def get_pictures(Xd, location):
    images = np.empty((Xd.shape[0], 227, 227, 3))
    #print(Xd)
    #print(location)
    for i in range(Xd.shape[0]):
        im = cv2.imread(location[i] + "/" + Xd[i])
        x = random.randint(0, 256 - 227)
        im = im[x:x+227, x:x+227]

        if random.random() < .5:
            im = cv2.flip(im, 0)
        #print(location[i] + "/" + Xd[i])
        images[i] = im
    assert not np.any(np.isnan(images))
    images = images - mean_image
    return images



data = pd.read_csv('227X227_v2.csv')

X_data = np.array(data['file_name'])
y_data = np.array(data['Letter'])
location = np.array(data['dir_name'])

label_encoder = LabelEncoder()
#'Part5', 'Part2', 'Part3', 'part4','Part1'
#'C', 'B', 'D', 'A', 'Part5', 'Part2', 'Part3', 'part4', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_9', 'user_10', 'Part1'
train_signers = []
test_signers = ['E']

X_train = data.loc[data['dir_name'].isin(train_signers)]['file_name'].values
y_train = data.loc[data['dir_name'].isin(train_signers)]['Letter'].values
y_train = label_encoder.fit_transform(y_train)
train_loc = data.loc[data['dir_name'].isin(train_signers)]['root'].values

X_val = data.loc[data['dir_name'].isin(test_signers)]['file_name'].values
y_val = data.loc[data['dir_name'].isin(test_signers)]['Letter'].values
y_val = label_encoder.fit_transform(y_val)
val_loc = data.loc[data['dir_name'].isin(test_signers)]['root'].values

print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

y_val = keras.utils.to_categorical(y_val, num_classes=24)

model = kit_model.KitModel(weight_file = 'auto_gen_from_tf/weights.npy')
model.compile(keras.optimizers.SGD(), loss='mean_squared_error', metrics=['accuracy'])

batch_size = 32
losses = []
accuracies = []
batches = 0

for i in range(int(math.ceil(X_val.shape[0] / batch_size))):
    start_idx = (i * batch_size) % X_val.shape[0]

    loss, accuracy = model.test_on_batch(get_pictures(X_val[start_idx:start_idx + batch_size], val_loc[start_idx:start_idx + batch_size]), y_val[start_idx:start_idx + batch_size])

    actual_batch_size = y_val[start_idx:start_idx + batch_size].shape[0]

    losses.append(loss * actual_batch_size)
    accuracies.append(accuracy)
    batches += 1

loss = np.sum(losses) / (X_val.shape[0])
accuracy = np.sum(accuracies) / batches

print(loss)
print(accuracy)















#
