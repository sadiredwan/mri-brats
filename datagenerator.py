from tensorflow import keras
import numpy as np
import os
import nibabel as nb
import cv2
import tensorflow as tf

IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, list_ids, dim=(IMG_SIZE,IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.__on_epoch_end__()
    
    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_ids = [self.list_ids[k] for k in indexes]
        X, y = self.__generate_data__(batch_ids)
        return X, y

    def __on_epoch_end__(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __generate_data__(self, batch_ids):
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))
        for c, i in enumerate(batch_ids):
            case_path = os.path.join(self.path, i)
            data_path = os.path.join(case_path, f'{i}_flair.nii.gz')
            flair = nb.load(data_path).get_fdata()
            data_path = os.path.join(case_path, f'{i}_t1ce.nii.gz')
            t1ce = nb.load(data_path).get_fdata()
            data_path = os.path.join(case_path, f'{i}_seg.nii.gz')
            seg = nb.load(data_path).get_fdata()
            for j in range(VOLUME_SLICES):
                X[j+VOLUME_SLICES*c, :, :, 0] = cv2.resize(flair[:, :, j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                X[j+VOLUME_SLICES*c, :, :, 1] = cv2.resize(t1ce[:, :, j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                y[j+VOLUME_SLICES*c] = seg[:, :, j+VOLUME_START_AT]
        y[y==4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
        return X/np.max(X), Y
