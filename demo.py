from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
from imutils.video import VideoStream, FileVideoStream
import argparse
import time
import numpy as np
import random
import pandas as pd
import tensorflow as tf

class PhotoBoothApp:
    def __init__(self, vs, height, width):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.fourcc = 'DIVX'
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.frameSize = (width, height) # video formats and sizes also depend and vary according to the camera used
        #self.video_cap = cv2.VideoCapture(self.device_index)

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.text = tki.Text(self.root, height=1)
        #self.text.config(font=("Georgia", 30))
        #self.text.config.grid(column=4, row=0, pady=20, ipadx=0)
        self.panel = None

        self.prediction = tki.Text(self.root, height=1)
        self.prediction.grid(column=4, row=3, pady=20, ipadx=0)
        self.confidence_scores = tki.Text(self.root, height=1)
        self.confidence_scores.grid(column=4, row=5, pady=20, ipadx=0)

        self.cap = None

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("ASL Fingerspelling Video Filmer")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    def resize(self, im):
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


    def analyze(self, image, sess, X, probs):

        #image = cv2.imread(img_pth)
        image = self.resize(image)
        x = random.randint(0, 256 - 227)
        resized_image = image[x:x+227, x:x+227]

        image = resized_image[np.newaxis, :, :, :]
        #print(image.shape)
        #start = time.time()
        predictions = np.argsort(sess.run([probs], feed_dict={X: image}))[::-1]
        #end = time.time()
        #print(end - start)
        return predictions

    def videoLoop(self):
        mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13,
                   'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x': 22, 'y': 23}

        int_to_char = {v: k for k, v in mapping.items()}

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

            with tf.device("/device:GPU:0"):  # "/cpu:0" or "/gpu:0"

                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

                    saver = tf.train.import_meta_graph("C:/Riley/DeepLetters/CNN/trained_networks/one opt/static_v2_lr-1e-06/epoch-7/static_v2_lr-1e-06.ckpt.meta")

                    saver.restore(sess=sess, save_path="C:/Riley/DeepLetters/CNN/trained_networks/one opt/static_v2_lr-1e-06/epoch-7/static_v2_lr-1e-06.ckpt")

                    graph = sess.graph
                    probs = graph.get_tensor_by_name('prob3:0')
                    probs = tf.contrib.layers.flatten(probs)

                    X = graph.get_tensor_by_name('inputs:0')

                    try:

                        #while not self.stopEvent.is_set():
                        while vs.more():
                            time.sleep(0.5)
                            self.frame = self.vs.read()
                            print("predicting")
                            predictions = self.analyze(self.frame, sess, X, probs)[0][0]
                            print(predictions)
                            self.frame = imutils.resize(self.frame)
                            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                            image = cv2.flip(image, 1)
                            image = Image.fromarray(image)
                            image = ImageTk.PhotoImage(image)

                            # if the panel is not None, we need to initialize it
                            if self.panel is None:
                                self.panel = tki.Label(image=image)
                                self.panel.image = image
                                #self.panel.pack(side="left", padx=10, pady=10)
                                self.panel.grid(column=0, row=0, rowspan=10, columnspan=3)

                            else:
                                self.prediction.delete('1.0', tki.END)
                                self.confidence_scores.delete('1.0', tki.END)

                                self.prediction.insert(tki.INSERT, int_to_char[predictions[0]])
                                self.confidence_scores.insert(tki.INSERT, [int_to_char[prediction] for prediction in predictions])
                                self.panel.configure(image=image)
                                self.panel.image = image

                    except ValueError:
                        print("[INFO] caught a RuntimeError")

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")

        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
        cv2.destroyAllWindows()
        vs.stream.release()


if __name__=='__main__':

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-o", "--output", required=True,
    # 	help="path to output directory to store snapshots")
    # ap.add_argument("-p", "--picamera", type=int, default=-1,
    # 	help="whether or not the Raspberry Pi camera should be used")
    # args = vars(ap.parse_args())

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] warming up camera...")
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("C:/Riley/DeepLetters/video_data/signer2/why.mp4")

    width = int(cap.get(3))  # float
    height = int(cap.get(4)) # float

    print("width: " + str(width))
    print("height: " + str(height))
    cap.release()

    time.sleep(2.0)

    #vs = VideoStream().start()
    vs = FileVideoStream("C:/Riley/DeepLetters/video_data/signer2/why.mp4").start()
    time.sleep(2.0)

    # start the app
    pba = PhotoBoothApp(vs, height=height, width=width)
    pba.root.mainloop()
