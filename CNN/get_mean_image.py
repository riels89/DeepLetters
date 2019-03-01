import numpy as np
import pandas as pd
import cv2
import random


def get_picture(Xd, root):
    images = np.empty((256, 256, 3))
    im = cv2.imread(str(root) +'/'+ str(Xd))
    print(str(root) +'/'+ str(Xd))
    images = im
    assert not np.any(np.isnan(images))
    return images

def get_pictures(Xd):
    images = np.empty((Xd.shape[0], 256, 256, 3))
    for i in range(Xd.shape[0]):
        im = cv2.imread('data_heap/' + Xd[i])
        images[i] = im
    assert not np.any(np.isnan(images))
    return images

data = pd.read_csv('227X227_v2.csv')

train_signers = ['C', 'B', 'A', 'D', 'Part5', 'Part2', 'Part3', 'part4']
test_signers = ['E','Part1']

X_data = data.loc[data['dir_name'].isin(train_signers)]['file_name'].values
root = data.loc[data['dir_name'].isin(train_signers)]['root'].values

sum = np.zeros([227, 227, 3])
for i in range(len(X_data)):
    print(X_data[i])
    image = get_picture(X_data[i], root[i])
    x = random.randint(0, 256 - 227)
    image = image[x:x+227, x:x+227]

    sum += image
mean_image = sum / len(X_data)
print(mean_image.shape)
#print(np.mean(get_pictures(X_data, root), axis=0).shape)
np.save('CNN/mean_image.npy', mean_image)
