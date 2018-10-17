import cv2
import numpy as np
import os
from scipy import ndimage, misc
import pandas as pd

def load_videos(folder_path, number_vids=-1, symbols=list('abcdefghijklmnopqrstuvwxyz1234567890')):
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

def make_data_heap():
    '''
    numbers
    {'a': 2816, 'b': 2868, 'c': 3056, 'd': 2820, 'e': 2821, 'f': 2755, 'g': 2819, 'h': 2836,
     'i': 2771, 'j': 70, 'k': 3079, 'l': 2894, 'm': 2850, 'n': 2834, 'o': 2796, 'p': 2943, 'q': 2812,
      'r': 3080, 's': 2924, 't': 2759, 'u': 2793, 'v': 2878, 'w': 3248, 'x': 2871, 'y': 2806, 'z': 70}
    '''
    letters = list('abcdefghijklmnopqrstuvwxyz')
    letter_counts = {letter: 0 for letter in letters}
    for root, dir, filenames in os.walk("C:/Users/riley/DeepLetters/Data/Dataset - Github sign language data (mon95)"):
        for file in filenames:
            filepath = os.path.join(root, file)
            image = cv2.imread(filepath)
            image_resized = cv2.resize(image, (227, 227))
            new_loc = os.path.join('C:/Users/riley/DeepLetters/data_heap', file[0].lower() + str(letter_counts[file[0].lower()])) + '.jpg'
            cv2.imwrite(new_loc, image_resized)
            letter_counts[file[0].lower()] += 1

    for actor in ('A', 'B', 'C', 'D', 'E'):
        for root, dir, filenames in os.walk("C:/Users/riley/DeepLetters/Data/dataset5/" + actor):
            for file in filenames:
                if not file.startswith('depth'):
                    filepath = os.path.join(root, file)
                    print(filepath)
                    image = cv2.imread(filepath)
                    image_resized = cv2.resize(image, (227, 227))
                    new_loc = os.path.join('C:/Users/riley/DeepLetters/data_heap',
                                           filepath[43].lower() + str(letter_counts[filepath[43].lower()])) + '.jpg'
                    print(new_loc)
                    cv2.imwrite(new_loc, image_resized)
                    letter_counts[filepath[43].lower()] += 1

    for part in ('Part1', 'Part2', 'Part3', 'Part4', 'Part5'):
        for root, dirnames, filenames in os.walk("C:/Users/riley/DeepLetters/Data/" + part):
            for file in filenames:
                if not file[6].isdigit():
                    filepath = os.path.join(root, file)
                    print(filepath)
                    image = cv2.imread(filepath)
                    image_resized = cv2.resize(image, (227, 227))
                    new_loc = os.path.join('C:/Users/riley/DeepLetters/data_heap', file[6].lower() + str(letter_counts[file[6].lower()])) + '.jpg'
                    print(new_loc)
                    cv2.imwrite(new_loc, image_resized)
                    letter_counts[file[6].lower()] += 1
    print(letter_counts)


def make_picture_list():
    picture_list = []
    for root, dirnames, filenames in os.walk('C:/Users/riley/DeepLetters/data_heap/'):
        for file in filenames:
            if file[0] != 'j' and file[0] != 'z':
                print(file)
                picture_list.append({'file_name': file, 'Letter': file[0], 'Number': file[1:len(file) - 4]})
    picture_list = pd.DataFrame(picture_list)
    print(picture_list)
    picture_list.to_csv('C:/Users/riley/DeepLetters/227X227.csv')

#make_data_heap()
make_picture_list()
# videos = load_data('C:/Users/Riley/Documents/.Research Project/0', 1, symbols='ABCDEFGHIJKLMNOPQRSTUVWYXZ')

# print(videos['A'][1][400, 400])
    # letters = list('abcdefghijklmnopqrstuvwxyz')
    # letter_counts = {letter: 0 for letter in letters}
    # for root, dir, filenames in os.walk("C:/Users/riley/DeepLetters/Data/Dataset - Github sign language data (mon95)"):
    #     for file in filenames:
    #         filepath = os.path.join(root, file)
    #         image = cv2.imread(filepath)
    #         image_resized = cv2.resize(image, (227, 227))
    #         new_loc = os.path.join('C:/Users/riley/DeepLetters/data_heap', file[0].lower() + str(letter_counts[file[0].lower()])) + '.jpg'
    #         cv2.imwrite(new_loc, image_resized)
    #         letter_counts[file[0].lower()] += 1
