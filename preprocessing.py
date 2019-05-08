import cv2
import numpy as np
import os
from scipy import ndimage, misc
import pandas as pd

def resize(im):
    desired_size = 227

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
    # for root, dir, filenames in os.walk("C:/Riley/DeepLetters/Data/Dataset - Github sign language data (mon95)"):
    #     letter_counts = {letter: 0 for letter in letters}
    #     for file in filenames:
    #         filepath = os.path.join(root, file)
    #         image = cv2.imread(filepath)
    #         image_resized = cv2.resize(image, (227, 227))
    #         new_loc = os.path.join('C:/Riley/DeepLetters/data_heap/dir', file[0].lower() + str(letter_counts[file[0].lower()])) + '.jpg'
    #         cv2.imwrite(new_loc, image_resized)
    #         letter_counts[file[0].lower()] += 1

    for actor in ('A', 'B', 'C', 'D', 'E'):
        letter_counts = {letter: 0 for letter in letters}
        os.mkdir('C:/Riley/DeepLetters/data_heap/' + actor)
        for root, dir, filenames in os.walk("C:/Riley/DeepLetters/Data/dataset5/" + actor):
            for file in filenames:
                if not file.startswith('depth'):
                    filepath = os.path.join(root, file)
                    print(filepath)
                    image = cv2.imread(filepath)
                    image_resized = resize(image)
                    new_loc = os.path.join('C:/Riley/DeepLetters/data_heap/' + actor,
                                           filepath.rsplit('\\',1)[0][-1].lower() + str(letter_counts[filepath.rsplit('\\',1)[0][-1].lower()])) + '.jpg'
                    print(new_loc)
                    cv2.imwrite(new_loc, image_resized)
                    letter_counts[filepath.rsplit('\\',1)[0][-1].lower()] += 1

    for part in ('Part1', 'Part2', 'Part3', 'Part4', 'Part5'):
        letter_counts = {letter: 0 for letter in letters}
        os.mkdir('C:/Riley/DeepLetters/data_heap/' + part)
        for root, dirnames, filenames in os.walk("C:/Riley/DeepLetters/Data/" + part):
            for file in filenames:
                if not file[6].isdigit():
                    filepath = os.path.join(root, file)
                    print(filepath)
                    image = cv2.imread(filepath)
                    image_resized = resize(image)
                    new_loc = os.path.join('C:/Riley/DeepLetters/data_heap/' + part, file[6].lower() + str(letter_counts[file[6].lower()])) + '.jpg'
                    print(new_loc)
                    cv2.imwrite(new_loc, image_resized)
                    letter_counts[file[6].lower()] += 1
    print(letter_counts)

    for part in ('user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_9', 'user_10'):
        letter_counts = {letter: 0 for letter in letters}
        os.mkdir('C:/Riley/DeepLetters/data_heap/' + part)
        for root, dirnames, filenames in os.walk("C:/Riley/DeepLetters/Data/Dataset - Github sign language data (mon95)/" + part):
            for file in filenames:
                    filepath = os.path.join(root, file)
                    print(filepath)
                    image = cv2.imread(filepath)
                    image_resized = resize(image)
                    new_loc = os.path.join('C:/Riley/DeepLetters/data_heap/' + part, file[0].lower() + str(letter_counts[file[0].lower()])) + '.jpg'
                    print(new_loc)
                    cv2.imwrite(new_loc, image_resized)
                    letter_counts[file[0].lower()] += 1
    print(letter_counts)

    for part in ('Signer_1', 'Signer__2', 'Signer_3'):
        letter_counts = {letter: 0 for letter in letters}
        os.mkdir('C:/Riley/DeepLetters/data_heap/' + part)
        for root, dirnames, filenames in os.walk("C:/Riley/DeepLetters/Data/Dataset/Garbage_Frames_Train" + part):
            for file in filenames:
                filepath = os.path.join(root, file)
                print(filepath)
                image = cv2.imread(filepath)
                image_resized = resize(image)
                new_loc = os.path.join('C:/Riley/DeepLetters/data_heap/' + part, file[0].lower() + str(letter_counts[file[0].lower()])) + '.jpg'
                print(new_loc)
                cv2.imwrite(new_loc, image_resized)
                letter_counts[file[0].lower()] += 1
    print(letter_counts)

    for part in ('Signer'):
        letter_counts = {letter: 0 for letter in letters}
        os.mkdir('C:/Riley/DeepLetters/data_heap/' + part)
        for root, dirnames, filenames in os.walk("C:/Riley/DeepLetters/Data/Dataset/Garbage_Frames_Validation" + part):
            for file in filenames:
                filepath = os.path.join(root, file)
                print(filepath)
                image = cv2.imread(filepath)
                image_resized = resize(image)
                new_loc = os.path.join('C:/Riley/DeepLetters/data_heap/' + part, file[0].lower() + str(letter_counts[file[0].lower()])) + '.jpg'
                print(new_loc)
                cv2.imwrite(new_loc, image_resized)
                letter_counts[file[0].lower()] += 1
    print(letter_counts)

def make_picture_list():
    gs = ['g_1', 'g_2', 'g_3']
    picture_list = []
    for root, dirnames, filenames in os.walk('C:/Riley/DeepLetters/data_heap'):
        for file in filenames:
            if file[0] != 'j' and file[0] != 'z':
                print(file)
                picture_list.append({'root': root, 'dir_name':root.rsplit('\\',1)[1], 'file_name': file, 'Letter': file[0], 'Number': file[1:len(file) - 4]})
    picture_list = pd.DataFrame(picture_list)
    print(picture_list)
    picture_list.to_csv('C:/Riley/DeepLetters/227X227_v2.csv')

def make_new_picture_list():
    new_picture_list = []

    for root, dirnames, filenames in os.walk('C:/Users/Riley/DeepLetters/New_Images_DeepLetters'):
        for file in filenames:
            if file[0] != 'j' and file[0] != 'z':
                print(file)
                print(file[-5])
                new_picture_list.append({'root': 'New_Images_DeepLetters', 'dir_name':'New_Images_DeepLetters', 'file_name': file, 'Letter': file[-5]})
    new_picture_list = pd.DataFrame(new_picture_list)
    print(new_picture_list)
    new_picture_list.to_csv('C:/Users/Riley/DeepLetters/227X227_new_images.csv')

def crop_img():
    new_picture_list = []
    for root, dirnames, filenames in os.walk('C:/Riley/DeepLetters/New_Images_DeepLetters'):
        for file in filenames:
            filepath = os.path.join(root, file)
            image = cv2.imread(filepath)
            image = image[400:2619, 804:2219]
            print(image.shape)
            image_resized = resize(image)
            # cv2.imshow("cropped", image_resized)
            # cv2.waitKey(0)

            print(image_resized.shape)
            cv2.imwrite("cropped_test/" + file, image_resized)
            #cropped_img = img[]


# make_data_heap()
# make_picture_list()
# make_new_picture_list()
crop_img()


# videos = load_data('C:/Riley/Documents/.Research Project/0', 1, symbols='ABCDEFGHIJKLMNOPQRSTUVWYXZ')

# print(videos['A'][1][400, 400])
    # letters = list('abcdefghijklmnopqrstuvwxyz')
    # letter_counts = {letter: 0 for letter in letters}
    # for root, dir, filenames in os.walk("C:/Riley/DeepLetters/Data/Dataset - Github sign language data (mon95)"):
    #     for file in filenames:
    #         filepath = os.path.join(root, file)
    #         image = cv2.imread(filepath)
    #         image_resized = cv2.resize(image, (227, 227))
    #         new_loc = os.path.join('C:/Riley/DeepLetters/data_heap_v4', file[0].lower() + str(letter_counts[file[0].lower()])) + '.jpg'
    #         cv2.imwrite(new_loc, image_resized)
    #         letter_counts[file[0].lower()] += 1
