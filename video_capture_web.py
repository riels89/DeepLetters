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

class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.fourcc = 'DIVX'
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.frameSize = (640, 480) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = "videos/temp_video.avi"
        #self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, 60, self.frameSize)

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        self.current_word = "test"
        self.recording = False
        self.writer = None

        # create a button, that when pressed, will take the current
        # frame and save it to file

        btn_take_start = tki.Button(self.root, text="Start",
                                   command=lambda: self.start())
        btn_take_start.pack(side="bottom", fill="both", expand="yes", padx=10,
                                         pady=10)
        btn_take_stop = tki.Button(self.root, text="Stop",
                                   command=lambda: self.onClose())
        btn_take_stop.pack(side="bottom", fill="both", expand="yes", padx=10,
                                         pady=10)


        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
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

                    self.video_out.write(np.array(self.frame.copy()))
                    # OpenCV represents images in BGR order; however PIL
                    # represents images in RGB order, so we need to swap
                    # the channels, then convert to PIL and ImageTk format
                    image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    # if the panel is not None, we need to initialize it
                    if self.panel is None:
                        self.panel = tki.Label(image=image)
                        self.panel.image = image
                        self.panel.pack(side="left", padx=10, pady=10)

                        # otherwise, simply update the panel
                    else:
                        self.panel.configure(image=image)
                        self.panel.image = image

            except ValueError:
                print("[INFO] caught a RuntimeError")

    def add_frame(self):
        cv2.VideoWriter.write(np.array(self.frame.copy()))

    def start(self):
        filename = self.current_word
        p = os.path.sep.join((self.outputPath, filename))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter()
        self.writer.open(p, fourcc, 60, (640,480))
        self.recording = True
        print("[INFO] saved {}".format(filename))
    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        #cv2.VideoCapture.release(self.video_out)
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store snapshots")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream().start()
time.sleep(2.0)

# start the app
pba = PhotoBoothApp(vs, args["output"])
pba.root.mainloop()
