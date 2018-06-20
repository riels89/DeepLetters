import cv2
import numpy as np


def load_data(folder_path, number_vids=-1, symbols=list('abcdefghijklmnopqrstuvwxyz1234567890')):
    videos = {}
    for i in range(number_vids):
        symbol = symbols[i]
        images = []
        video = cv2.VideoCapture(folder_path + "/" + symbol + ".mp4")
        success = True
        while success:
            success, frame = video.read()
            if success:
                images.append(np.array(frame))
        videos[symbol] = images

    video.release()
    cv2.destroyAllWindows()
    return videos


videos = load_data('C:/Users/Riley/Documents/.Research Project/0', 1, symbols='ABCDEFGHIJKLMNOPQRSTUVWYXZ')

print(videos['A'][1][400, 400])
