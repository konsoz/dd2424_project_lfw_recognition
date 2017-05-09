#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path
import numpy as np
import tensorflow as tf
import logging as log
import pickle as pickle
import gzip
from timeit import default_timer as timer

IMG_SIZE = 250
RE_IMG_SIZE = 96
NUM_LABELS =  19

"""
Count total number of images
"""


def getNumImages(image_dir):
    count = 0
    for dirName, subdirList, fileList in os.walk(image_dir):
        for img in fileList:
            count += 1
    return count


"""
Return the dataset as images and labels
"""

def convertDataset(image_dir):
    THRESHOLD = 10  # TODO

    # TODO create function for this
    # Calculatge number of labels
    global num_labels
    num_labels = 0
    for dirName in os.listdir(image_dir):
        path = os.path.join(image_dir, dirName)
        if (len(os.listdir(path)) < THRESHOLD):
            continue
        else:
            num_labels += 1

    print("SAVE THIS, NUM LABELS IS: %d" % num_labels)
    global labels_dict
    labels_dict = {}
    label = np.eye(num_labels)  # Convert labels to one-hot-vector
    i = 0

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    log.info("Start processing images (Dataset.py) ")
    start = timer()
    for dirName in os.listdir(image_dir):
        path = os.path.join(image_dir, dirName)
        if (len(os.listdir(path)) < THRESHOLD):
            continue

        labels_dict[dirName] = i
        label_i = label[i]
        print("NOW AT %d/%d %fpercent" % (i + 1, num_labels, float(i + 1) / num_labels * 100))
        i += 1
        # log.info("Execution time of convLabels function = %.4f sec" % (end1-start1))

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            if os.path.isfile(img_path) and img.endswith('jpg'):
                img_bytes = tf.read_file(img_path)
                img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
                yield img_u8.eval(session=session), label_i
    end = timer()
    log.info("End processing images (Dataset.py) - Time = %.2f sec" % (end - start))


def saveDataset(image_dir, file_path):
    with gzip.open(file_path, 'wb') as file:
        for img, label in convertDataset(image_dir):
            pickle.dump((img, label), file)


def split_list(a_list, ratio):
    index = int(len(a_list)*ratio)
    return a_list[:index], a_list[index:]

def convertDataset2(image_dir):
    def read_and_encode_files(file_list, file_list_label, result_list):
        for img in file_list:
            img_path = os.path.join(path, img)
            if  not os.path.isfile(img_path) and not img.endswith('jpg'):
                continue
            img_bytes = tf.read_file(img_path)
            img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
            #img_padded_or_cropped = tf.image.resize_image_with_crop_or_pad(img_u8, IMG_SIZE, IMG_SIZE)
            result_list.append((img_u8.eval(session=session), file_list_label,))

    THRESHOLD = 40  # TODO
    dataset_train = []
    dataset_valid = []
    dataset_test = []

    # TODO create function for this
    # Calculatge number of labels
    global num_labels
    num_labels = 0
    for dirName in os.listdir(image_dir):
        path = os.path.join(image_dir, dirName)
        if (len(os.listdir(path)) < THRESHOLD):
            continue
        else:
            num_labels += 1

    print("SAVE THIS, NUM LABELS IS: %d" % num_labels)
    global labels_dict
    labels_dict = {}
    label = np.eye(num_labels)  # Convert labels to one-hot-vector
    i = 0

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    log.info("Start processing images (Dataset.py) ")
    start = timer()
    for dirName in os.listdir(image_dir):
        path = os.path.join(image_dir, dirName)
        if (len(os.listdir(path)) < THRESHOLD):
            continue

        labels_dict[dirName] = i
        label_i = label[i]
        print("NOW AT %d/%d %fpercent" % (i + 1, num_labels, float(i + 1) / num_labels * 100))
        i += 1
        # log.info("Execution time of convLabels function = %.4f sec" % (end1-start1))

        file_list = os.listdir(path)
        file_list_train, file_list_2 = split_list(file_list, 6/10)
        file_list_valid, file_list_test = split_list(file_list_2, 1/2)

        read_and_encode_files(file_list_train, label_i, dataset_train)
        read_and_encode_files(file_list_valid, label_i, dataset_valid)
        read_and_encode_files(file_list_test, label_i, dataset_test)

    end = timer()
    log.info("End processing images (Dataset.py) - Time = %.2f sec" % (end - start))

    return dataset_train, dataset_valid, dataset_test


def saveDataset2(image_dir, file_path):
    dataset_train, dataset_valid, dataset_test = convertDataset2(image_dir)
    with gzip.open(file_path+'_train', 'wb') as file:
        for img, label in dataset_train:
            pickle.dump((img, label), file)
    with gzip.open(file_path+'_valid', 'wb') as file:
        for img, label in dataset_valid:
            pickle.dump((img, label), file)
    with gzip.open(file_path+'_test', 'wb') as file:
        for img, label in dataset_test:
            pickle.dump((img, label), file)


def loadDataset(file_path):
    with gzip.open(file_path) as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break

def saveShuffle(l, file_path='images_shuffled.pkl'):
    with gzip.open(file_path, 'wb') as file:
        for img, label in l:
            pickle.dump((img, label), file)
