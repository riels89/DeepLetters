import cv2
import numpy as np
import os
import pandas as pd
import random

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



data = pd.read_csv("ChicagoFSWild/ChicagoFSWild.csv")
data['np_path'] = None
length = len(data.index)
root = "./ChicagoFSWild/videos/"
save_root = "./video_data/"
mean_image = np.load('CNN/mean_image.npy')

for index, row in data.iloc[3845:].iterrows():
    save_path = save_root + "singer" + str(row["signer"] + 4) + "/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = save_path + row["label_proc"] + ".npy"
    
    path = root + row["filename"] + "/"
    video = np.zeros(shape=(row['number_of_frames'], 256, 256, 3))
    for frame in range(row['number_of_frames']):
        im_path = path + '{:04}'.format(frame + 1) + ".jpg"
        
        im =  cv2.imread(im_path)
        resized_image = resize(im)
        video[frame] = resized_image

    np.save(save_path, video)
    row["np_path"] = save_path
    print(video.shape)
    print(str(index+1) +"/"+ str(length))
    
data.to_csv('ChicagoFSWild_with_np.csv')

    
    
    
    
    
    
    
    
    
    
    
    
    