import os

import numpy as np
from torch.utils.data import DataLoader

from datasets.datasets import Dataset
from datasets.unaligned_data_loader import UnalignedDataLoader
from utils import FileIO


def save_pseudo_dataset(data_target, leave_one_num=0, dataset='ENABL3S', sensor_num=0):
    folder_path = f'data/{dataset}/pseudo_dataset'
    os.makedirs(folder_path, exist_ok=True)
    file_path = '{}/{}_sensor_{}.npy'.format(
        folder_path, leave_one_num, sensor_num)
    np.save(file_path, data_target)
    print('Save pseudo dataset to {}'.format(file_path))


def load_target_data(leave_one_num=0, dataset='ENABL3S', sensor_num=0, is_resize=True):
    if 'ENABL3S' == dataset:
        x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_ENABL3S_mat(data_path='data/ENABL3S/AB_', X_dim=4,
                                    is_resize=is_resize,
                                    leave_one_num=leave_one_num,
                                    sensor_num=sensor_num)
    elif 'DSADS' == dataset:
        x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_DSADS_mat(data_path='data/DSADS/',
                                  feature_length=6 * 45, X_dim=4,
                                  is_resize=is_resize, leave_one_num=leave_one_num,
                                  sensor_num=sensor_num)
    # In our cross-subject adaptation experiments, there is no validation set to determine the early-stop time.
    x_t_test = np.concatenate([x_t_val, x_t_test], axis=0)
    y_t_test = np.concatenate([y_t_val, y_t_test], axis=0)
    dataset_target = {'x_t_train': x_t_train, 'y_t_train': y_t_train,
                      'x_t_test': x_t_test, 'y_t_test': y_t_test}
    return dataset_target


def read_pseudo_dataset(batch_size, is_resize=False,
                        leave_one_num=-1, dataset='ENABL3S',
                        sensor_num=0):
    '''
    dataset_target = {'x_t_train': x_t_train, 'y_t_train': y_t_train, 
                      'y_pseudo_t_train': y_pseudo_t_train,
                      'x_t_test': x_t_test, 'y_t_test': y_t_test}
    '''
    if 'ENABL3S' == dataset:
        folder_path = 'data/ENABL3S/pseudo_dataset'
        os.makedirs(folder_path, exist_ok=True)
    elif 'DSADS' == dataset:
        folder_path = 'data/DSADS/pseudo_dataset'
        os.makedirs(folder_path, exist_ok=True)
    file_path = '{}/{}_sensor_{}.npy'.format(
        folder_path, leave_one_num, sensor_num)
    dataset_target = np.load(file_path, allow_pickle=True).item()
    x_t_train, y_pseudo_t_train, x_t_test, y_t_test = \
        dataset_target['x_t_train'], dataset_target['y_pseudo_t_train'], \
        dataset_target['x_t_test'], dataset_target['y_t_test']

    data_train = DataLoader(
        Dataset(x_t_train, y_pseudo_t_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    data_test = DataLoader(
        Dataset(x_t_test, y_t_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    training_set_shape = y_pseudo_t_train.shape[0]
    test_set_shape = y_t_test.shape[0]
    print(f'Train shape: {training_set_shape}, Test shape {test_set_shape}')
    return data_train, data_test


def dataset_read(batch_size, is_resize=False,
                 leave_one_num=-1, dataset='ENABL3S',
                 sensor_num=0):
    S_train = {}
    S_val = {}
    T_train = {}
    T_val = {}

    S_test = {}
    T_test = {}

    if 'ENABL3S' == dataset:
        x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_ENABL3S_mat(data_path='data/ENABL3S/AB_', X_dim=4,
                                    is_resize=is_resize,
                                    leave_one_num=leave_one_num,
                                    sensor_num=sensor_num)
    elif 'DSADS' == dataset:
        x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_DSADS_mat(data_path='data/DSADS/',
                                  feature_length=6*45, X_dim=4,
                                  is_resize=is_resize, leave_one_num=leave_one_num,
                                  sensor_num=sensor_num)

    S_train['imgs'] = x_s_train
    S_train['labels'] = y_s_train
    T_train['imgs'] = x_t_train
    T_train['labels'] = y_t_train

    # input target samples for both
    S_val['imgs'] = np.r_[x_s_val]
    S_val['labels'] = np.r_[y_s_val]
    T_val['imgs'] = np.r_[x_t_val]
    T_val['labels'] = np.r_[y_t_val]

    S_test['imgs'] = x_s_test
    S_test['labels'] = y_s_test
    T_test['imgs'] = x_t_test
    T_test['labels'] = y_t_test

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S_train, T_train, batch_size, batch_size)
    data_train = train_loader.load_data()

    validation_loader = UnalignedDataLoader()
    validation_loader.initialize(S_val, T_val, batch_size, batch_size)
    data_val = validation_loader.load_data()

    final_test_loader = UnalignedDataLoader()
    final_test_loader.initialize(S_test, T_test, batch_size, batch_size)
    data_test = final_test_loader.load_data()
    print('Train shape: {}, Validation shape {}, Test shape {}'.format(T_train['labels'].shape[0],
                                                                       T_val['labels'].shape[0],
                                                                       T_test['labels'].shape[0]))
    # print('Target validation shape: {}'.format(T_val['labels'].shape[0] + T_test['labels'].shape[0]))
    return data_train, data_val, data_test
