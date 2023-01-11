import os
import numpy as np
import random
from tqdm import tqdm
import cv2
import sys


def add_peppersalt_noise(image, n=1000):  # 生成椒盐噪声
    result = image.copy()
    w, h = image.shape[:2]
    # 生成n个椒盐噪声
    for i in range(n):
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        if np.random.randint(0, 2) == 0:
            result[x, y] = 0
        else:
            result[x, y] = 255
    return result


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def Preprocessing(image):
    gauss = cv2.GaussianBlur(image, (7, 7), 0)
    # gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
    # blur = cv2.blur(gauss, (3, 3))
    # cv2.imshow('gray', gray)
    # cv2.waitKey(10000)
    return gauss


datasets_path = "../init_data/train/datasets"

types_name = os.listdir(datasets_path)
types_name = sorted(types_name)
# print(types_name)

list_file = open('../init_data/train/cls_train.txt', 'w')
train_file_bar = tqdm(types_name, file=sys.stdout)
for cls_id, type_name in enumerate(train_file_bar):
    photos_path = os.path.join(datasets_path, type_name)
    if not os.path.isdir(photos_path):
        continue
    photos_name = os.listdir(photos_path)
    # print(photos_name)
    change_files, _ = data_split(photos_name, 0.2)
    # print(change_files)
    for change_file in change_files:
        img = cv2.imread('../init_data/train/datasets/' + type_name + '/' + change_file)
        img = add_peppersalt_noise(img)
        img = Preprocessing(img)
        cv2.imwrite('../init_data/train/datasets/' + type_name + '/' + change_file, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(1000)
    for photo_name in photos_name:
        list_file.write(
            str(cls_id) + ";" + '%s' % (os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
        list_file.write('\n')
list_file.close()
