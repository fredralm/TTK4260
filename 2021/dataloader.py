#!usr/bin/env Python
import os
import scipy
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Dict, Tuple

from utilities import remove_dict_keys
from preprocess import subset_selection

class Dataloader():
    def __init__(self, dir_path: str, data_file: str, cali_file: str, labels_file: str):
        ''' Dataloader class for the Indian Pines hyperspectral image dataset.
        arg dir_path: path to dataset directory
        arg data_file: name of the data file
        arg cali_file: name of the calibration file
        arg labels_file: name of the labels file 
        return: IndianPinesDataloader object'''
        # File paths
        data_path = dir_path + '/' + data_file
        cali_path = dir_path + '/' + cali_file
        labels_path = dir_path + '/' + labels_file
        assert os.path.exists(dir_path), 'Folder does not exist: {}'.format(
                os.path.abspath(dir_path))
        assert os.path.isfile(data_path), 'File does not exist: {}'.format(
                os.path.abspath(data_path))
        assert os.path.isfile(cali_path), 'File does not exist: {}'.format(
                os.path.abspath(cali_path))
        assert os.path.isfile(labels_path), 'File does not exist: {}'.format(
                os.path.abspath(labels_path))
        paths = {'data': data_path, 'calibration': cali_path, 'labels': labels_path}
        self._paths = paths
        # Load files
        mat_headers = ['__header__', '__version__', '__globals__']
        samples = remove_dict_keys(scipy.io.loadmat(data_path), mat_headers)['data']
        cali = remove_dict_keys(scipy.io.loadmat(cali_path), mat_headers)
        labels = remove_dict_keys(scipy.io.loadmat(labels_path), mat_headers)['labels']
        self._samples = samples
        self._calibration = cali
        self._labels = labels

    def get_samples(self):
        return self._samples.copy()

    def get_calibration(self, key: str=None) -> Dict:
        if key in self._calibration:
            return self._calibration[key]
        elif key == None:
            return self._calibration
        else:
            return None

    def get_labels(self) -> np.ndarray:
        return self._labels.copy()

    def get_wave_lengths(self, indices: np.ndarray=np.array([], dtype=int)):
        wls = np.squeeze(self.get_calibration('centers'))
        if indices.size != 0:
            wls = np.squeeze(subset_selection(wls[np.newaxis,:], indices))
        return wls

    def get_calibrated_samples(self):
        # Rawdata
        samples = self.get_samples().astype(float)
        cali = self.get_calibration()
        offset = cali['offset'][0,0]
        scale = cali['scale'][0,0]
        centers = np.squeeze(cali['centers'])
        # Calibration, change of units, non-negativity and radiance
        spec_rads = (samples - offset) / scale
        spec_rads /= (0.01)**(-2)
        spec_rads[spec_rads<0] = 0
        rads = spec_rads * centers
        return rads

def example_dataloader():
    dataloader = Dataloader('../datasets/indian_pines', 
            'indian_pines_corrected.mat', 'calibration_corrected.mat', 'indian_pines_gt.mat')

if __name__ == '__main__':
    example_dataloader()
