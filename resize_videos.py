import tensorflow as tf
import logging
import numpy as np
import cv2
import sys
import os
import random
import pandas as pd
from pprint import pprint
sys.path.append('CNN')

log = logging.getLogger()
log.setLevel(logging.DEBUG)

mean_image = np.load('CNN/mean_image.npy')

def resize(im):
    desired_size = 256

    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im

def get_pictures(video):

    if not os.path.isdir('video_data/' + video['filename'].split('/')[0]):
        os.mkdir('video_data/' + video['filename'].split('/')[0])
    writer = cv2.VideoWriter('video_data/' + video['filename'] + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (256, 256), True)
    print('video_data/' + video['filename'] + '.mp4')
    #video = np.empty([video['number_of_frames'], 227, 227, 3])
    fliped = random.random() < .5
    for i in range(video['number_of_frames']):
        print('ChicagoFS/videos/' + video['filename'] + '/' + str(i).zfill(4) + '.jpg')
        frame = cv2.imread('ChicagoFS/videos/' + video['filename'] + '/' + str(i+1).zfill(4) + '.jpg')
        resized_image = resize(frame)

        writer.write(resized_image)
    writer.release()

    cv2.destroyAllWindows()

data = pd.read_csv('ChicagoFS/ChicagoFSWild - OG backup.csv')

for index, row in data.iterrows():
    get_pictures(row)









#
