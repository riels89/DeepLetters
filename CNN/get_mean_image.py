import numpy as np
import pandas as pd
import cv2
def get_picture(Xd):
    images = np.empty((227, 227, 3))
    im = cv2.imread('data_heap/' + str(Xd))
    print('data_heap/' + str(Xd))
    images = im
    assert not np.any(np.isnan(images))
    return images

def get_pictures(Xd):
    images = np.empty((Xd.shape[0], 227, 227, 3))
    for i in range(Xd.shape[0]):
        im = cv2.imread('data_heap/' + Xd[i])
        images[i] = im
    assert not np.any(np.isnan(images))
    return images

data = pd.read_csv('227X227.csv')
X_data = np.array(data['file_name'])

sum = np.zeros([227, 227, 3])
for i in range(len(X_data)):
    print(X_data[i])
    image = get_picture(X_data[i])
    sum += image
mean_image = sum / len(X_data)
print(mean_image.shape)
print(np.mean(get_pictures(X_data[:1000]), axis=0).shape)
np.save('CNN/mean_image.npy', mean_image)
