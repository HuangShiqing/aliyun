# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import os



def load_image_path(data_path, valid_proportion, test_proportion, pos_path="1/", neg_path="0/"):
    """
    Args:
        data_path: list,数据所在文件夹，最后有一杠
        valid_proportion: float，验证集所占百分比，小数，如0.1
        test_proportion: float，测试集所占百分比
        pos_path: str,正样本所在文件夹
        neg_path: str,负样本所在文件夹
    Returns:
        x_train: list，dtype=str，图片路径
        y_train: list, dtype=int
        x_valid: list，dtype=str，图片路径
        y_valid: list, dtype=int
        x_test: list，dtype=str，图片路径
        y_test: list, dtype=int
    """

    pos_image_path = []
    pos_labels = []

    neg_image_path = []
    neg_labels = []

    ful_image_path = []
    ful_labels = []

    np.random.seed(0)

    pos_path = data_path + pos_path
    for img in tf.gfile.ListDirectory(pos_path):
        label = 1

        path = os.path.join(pos_path, img)
        pos_image_path.append(path)
        pos_labels.append(label)

    neg_path = data_path + neg_path
    for img in tf.gfile.ListDirectory(neg_path):
        label = 0

        path = os.path.join(neg_path, img)
        neg_image_path.append(path)
        neg_labels.append(label)

    ful_image_path = pos_image_path + neg_image_path
    ful_labels = pos_labels + neg_labels

    temp = np.array([ful_image_path, ful_labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    ful_image_path = list(temp[:, 0])
    ful_labels = list(temp[:, 1])
    ful_labels = [int(i) for i in ful_labels]

    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    from sklearn.model_selection import train_test_split
    if not valid_proportion == 0:
        x_train, x_valid, y_train, y_valid = train_test_split(ful_image_path, ful_labels,
                                                              test_size=(valid_proportion + test_proportion),
                                                              stratify=ful_labels, random_state=1)
        if not test_proportion == 0:
            x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=test_proportion / (
                    valid_proportion + test_proportion), stratify=y_valid, random_state=1)
    else:
        x_train = ful_image_path
        y_train = ful_labels

    print("train_num: %d ,pos_num: %d , neg_num: %d" % (
        len(y_train), y_train.count(1), len(y_train) - y_train.count(1)))
    print("valid_num: %d ,pos_num: %d , neg_num: %d" % (
        len(y_valid), y_valid.count(1), len(y_valid) - y_valid.count(1)))
    print("test_num : %d ,pos_num: %d , neg_num: %d" % (
        len(y_test), y_test.count(1), len(y_test) - y_test.count(1)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_batch(image, label, image_W, image_H, batch_size, capacity=500):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_images(image, [image_W, image_H], method=0)
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size])

    # label_batch = to_categorical(y=label_batch, nb_classes=2)
    return image_batch, label_batch


def count_pos(lables):
    num = 0
    for i in range(len(lables)):
        if lables[i] == 1:
            num = num + 1
    return num
