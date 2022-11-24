
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:30:35 2019

@author: kuangen
"""
import copy
from random import shuffle

import mat4py as m4p
import numpy as np
from sklearn import preprocessing


def load_DSADS_mat(data_path='data/1_dataset_UCI_DSADS/Raw/', X_dim=4,
                   is_one_hot=False, is_normalized=False,
                   is_resize=False, leave_one_num=-1, sub_num=8,
                   feature_length=5625, sensor_num=0):

    idx_vec = list(range(sub_num))
    if -1 == leave_one_num:
        shuffle(idx_vec)
        idx_train = idx_vec[:5]
        idx_test = idx_vec[5:-1]
    else:
        idx_test = [copy.deepcopy(idx_vec[leave_one_num])]
        idx_vec.pop(leave_one_num)
        idx_train = idx_vec

    # dataset:
    # x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
    # x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
    dataset = []
    for i in range(6):
        dataset.append(
            np.array([], dtype=np.float32).reshape(0, feature_length))
        dataset.append(np.array([], dtype=np.float32).reshape(0))
    for idx in idx_train:
        data_read = load_one_AB_mat(data_path, idx=idx)
        for j in range(6):
            dataset[j] = np.concatenate((dataset[j], data_read[j]), axis=0)
    for idx in idx_test:
        data_read = load_one_AB_mat(data_path, idx=idx)
        for j in range(6):
            dataset[j +
                    6] = np.concatenate((dataset[j + 6], data_read[j]), axis=0)
    for i in range(6):
        if is_resize:
            dataset[2 * i] = dataset[2 * i].reshape((-1,
                                                     45, int(feature_length / 45)))
        if 0 != sensor_num:
            dataset[2 * i] = dataset[2 * i][:, 9 *
                                            (sensor_num-1):9 * sensor_num, :]
        if is_normalized:
            dataset[2 * i] = preprocessing.scale(dataset[2 * i])
        for j in range(X_dim - len(dataset[2 * i].shape)):
            dataset[2 * i] = np.expand_dims(dataset[2 * i], axis=1)
        if is_one_hot:
            dataset[2 * i + 1] = one_hot(dataset[2 * i + 1],
                                         n_classes=int(1 + np.max(dataset[2 * i + 1])))

    return tuple(dataset)


def one_hot(y, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y = y.reshape(len(y))
    return np.eye(n_classes)[np.array(y, dtype=np.int32)]  # Returns FLOATS


def resize_feature(x, data_path):
    data = m4p.loadmat(data_path + 'idx.mat')
    idx_mat = np.array(data['idx_mat']) + 1
    zero_vec = np.zeros((len(x), 1))
    x_concat = np.concatenate((zero_vec, x), axis=1)
    return x_concat[:, idx_mat]


def load_ENABL3S_mat(data_path='0_dataset/AB_dataset/AB_', X_dim=4,
                     is_one_hot=False, is_normalized=False,
                     is_resize=False, leave_one_num=-1,
                     sensor_num=0):

    idx_vec = list(range(10))
    if -1 == leave_one_num:
        shuffle(idx_vec)
        idx_train = idx_vec[:5]
        idx_test = idx_vec[5:-1]
    else:
        idx_test = [copy.deepcopy(idx_vec[leave_one_num])]
        idx_vec.pop(leave_one_num)
        idx_train = idx_vec

    dataset = []
    for i in range(6):
        dataset.append(np.array([], dtype=np.float32).reshape(0, 368))
        dataset.append(np.array([], dtype=np.float32).reshape(0))
    for idx in idx_train:
        data_read = load_one_AB_mat(data_path, idx=idx)
        for j in range(6):
            dataset[j] = np.concatenate((dataset[j], data_read[j]), axis=0)
    for idx in idx_test:
        data_read = load_one_AB_mat(data_path, idx=idx)
        for j in range(6):
            dataset[j +
                    6] = np.concatenate((dataset[j + 6], data_read[j]), axis=0)
    for i in range(6):
        if is_resize:
            dataset[2 * i] = resize_feature(dataset[2 * i], data_path)
        if 0 != sensor_num:
            emg_idx = np.r_[np.arange(1, 4), np.arange(8, 12), np.arange(21, 25),
                            np.arange(29, 32)]
            imu_idx = np.r_[np.arange(4, 7), np.arange(
                12, 21), np.arange(26, 29)]
            angle_idx = np.r_[0, 7, 25, 32]
            sensor_idx = [emg_idx, imu_idx, angle_idx, np.r_[emg_idx, imu_idx],
                          np.r_[emg_idx, angle_idx], np.r_[imu_idx, angle_idx]]
            dataset[2 * i] = dataset[2 * i][:, sensor_idx[sensor_num-1], :]
        if is_normalized:
            dataset[2 * i] = preprocessing.scale(dataset[2 * i])
        for j in range(X_dim - len(dataset[2 * i].shape)):
            dataset[2 * i] = np.expand_dims(dataset[2 * i], axis=1)
        if is_one_hot:
            dataset[2 * i + 1] = one_hot(dataset[2 * i + 1],
                                         n_classes=int(1 + np.max(dataset[2 * i + 1])))

    return tuple(dataset)


def load_one_AB_mat(data_path='0_dataset/AB_dataset/AB_', idx=0):
    data = m4p.loadmat(data_path + str(idx) + '.mat')
    return [np.array(data['x_train']), np.array(data['y_train']),
            np.array(data['x_val']), np.array(data['y_val']),
            np.array(data['x_test']), np.array(data['y_test'])]


def load_mat(data_name, X_dim=4, is_one_hot=True, is_normalized=False):
    data = m4p.loadmat(data_name)
    data_array = [np.array(data['x_s_train']), np.array(data['y_s_train']),
                  np.array(data['x_s_val']), np.array(data['y_s_val']),
                  np.array(data['x_s_test']), np.array(data['y_s_test']),
                  np.array(data['x_t_train']), np.array(data['y_t_train']),
                  np.array(data['x_t_val']), np.array(data['y_t_val']),
                  np.array(data['x_t_test']), np.array(data['y_t_test'])]
    for i in range(6):
        if is_normalized:
            data_array[2 * i] = preprocessing.scale(data_array[2 * i])
        for j in range(X_dim - len(data_array[2 * i].shape)):
            data_array[2 * i] = np.expand_dims(data_array[2 * i], axis=-1)
        if is_one_hot:
            data_array[2 * i + 1] = one_hot(data_array[2 * i + 1],
                                            n_classes=int(1 + np.max(data_array[2 * i + 1])))
    return tuple(data_array)
