from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
from imutils.video import VideoStream
import argparse
import time
import numpy as np
import random
import pandas as pd

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
        self.words = None
        #self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = None

        self.words = pd.read_csv("ResearchWordsDeepLetters.csv")

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.text = tki.Text(self.root, height=1)
        #self.text.config(font=("Georgia", 30))
        #self.text.config.grid(column=4, row=0, pady=20, ipadx=0)
        self.text.grid(column=4, row=0, pady=20, ipadx=0)

        self.error_text = tki.Text(self.root, height=1)
        self.error_text.grid(column=4, row=7, pady=20, ipadx=0)
        self.panel = None

        self.filename = None
        self.current_word = "test"
        self.is_replaying = False
        self.cap = None
        self.recording = False
        self.writer = None

        # create a button, that when pressed, will take the current
        # frame and save it to file

        self.btn_take_start = tki.Button(self.root, text="Start Recording",
                                   command=lambda: self.start())
        #btn_take_start.pack(side="right", fill="both", expand="yes", padx=10, pady=10)
        self.btn_take_start.grid(column=4, row=1)
        self.btn_take_start.configure(bg = "Red")

        btn_take_stop = tki.Button(self.root, text="Stop Recording",
                                   command=lambda: self.stop())
        #btn_take_stop.pack(side="right", fill="both", expand="yes", padx=10, pady=10)
        btn_take_stop.grid(column=4, row=2)

        btn_take_save = tki.Button(self.root, text="Save",
                                   command=lambda: self.save())
        #btn_take_save.pack(side="right", fill="both", expand="yes", padx=10,pady=10)
        btn_take_save.grid(column=4, row=3)

        btn_take_next = tki.Button(self.root, text="Next Word",
                                   command=lambda: self.next_word())
        #btn_take_next.pack(side="right", fill="both", expand="yes", padx=10,pady=10)
        btn_take_next.grid(column=4, row=4)

        btn_take_replay = tki.Button(self.root, text="Replay",
                                   command=lambda: self.replay())
        #btn_take_next.pack(side="right", fill="both", expand="yes", padx=10,pady=10)
        btn_take_replay.grid(column=4, row=5)

        btn_take_delete = tki.Button(self.root, text="Delete",
                                   command=lambda: self.delete())
        #btn_take_next.pack(side="right", fill="both", expand="yes", padx=10,pady=10)
        btn_take_delete.grid(column=4, row=6)


        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("ASL Fingerspelling Video Filmer")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    def videoLoop(self):
            # DISCLAIMER:
            # I'm not a GUI developer, nor do I even pretend to be. This
            # try/except statement is a pretty ugly hack to get around
            # a RunTime error that Tkinter throws due to threading
            try:
                # keep looping over frames until we are instructed to stop
                while not self.stopEvent.is_set():
                    # grab the frame from the video stream and resize it to
                    # have a maximum width of 300 pixels
                    self.frame = self.vs.read()
                    self.frame = imutils.resize(self.frame)

                    if self.recording:
                        self.video_out.write(np.array(self.frame.copy()))
                    # OpenCV represents images in BGR order; however PIL
                    # represents images in RGB order, so we need to swap
                    # the channels, then convert to PIL and ImageTk format
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

                        # otherwise, simply update the panel
                    elif self.is_replaying:
                        ret, self.frame = self.cap.read()
                        if ret:
                            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                            image = cv2.flip(image, 1)
                            image = Image.fromarray(image)
                            image = ImageTk.PhotoImage(image)

                            self.panel.configure(image=image)
                            self.panel.image = image
                        else:
                            print("Replay done")
                            self.is_replaying = False
                            self.cap.release()
                    else:
                        self.panel.configure(image=image)
                        self.panel.image = image

            except ValueError:
                print("[INFO] caught a RuntimeError")

    def add_frame(self):
        cv2.VideoWriter.write(np.array(self.frame.copy()))
    def next_word(self):
        self.current_word = self.words["words"][random.randint(0, 124)]

        self.text.delete('1.0', tki.END)

        self.text.insert(tki.INSERT, "Your word is: " + self.current_word)
        #self.text.tag_add("here", "1.0", "1.4")

    def start(self):
        self.filename = "videos/" + self.current_word + ".mp4"
        self.video_out = cv2.VideoWriter(self.filename, self.video_writer, 60, self.frameSize)
        self.recording = True
        self.btn_take_start.configure(bg = "Green")
        print("Recording started ")

    def stop(self):
        self.btn_take_start.configure(bg = "Red")
        self.recording = False
        print("Recording stoped")

    def save(self):
        print(int(self.video_out.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.stop()
        self.video_out.release()

    def replay(self):
        if not self.recording:
            self.cap = cv2.VideoCapture(self.filename)
            self.is_replaying = True
            print("Set to replay")
        else:
            self.save()
            self.delete()
            self.error_text.delete('1.0', tki.END)
            self.error_text.insert(tki.INSERT, "You can't replay while recording!")

    def delete(self):
        self.save()
        os.remove(self.filename)

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
    cap = cv2.VideoCapture(0)
    width = cap.get(3)  # float
    height = cap.get(4) # float

    cap.release()

    time.sleep(2.0)

    vs = VideoStream().start()
    time.sleep(2.0)

    # start the app
    pba = PhotoBoothApp(vs, height=height, width=width)
    pba.root.mainloop()
