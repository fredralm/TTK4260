#!usr/bin/env Python
import os
import scipy
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition

from typing import Tuple, Dict


def data_preprocess(X_train: np.ndarray, X_test: np.ndarray, w: int, subset_inds: np.ndarray,
        do_smoothing: bool, do_subset: bool, do_snv: bool, do_normalize: bool):
    # Smoothing
    if do_smoothing:
        X_train = moving_average(X_train, w)
        X_test = moving_average(X_test, w)
    # Subset selection
    if do_subset:
        X_train = subset_selection(X_train, subset_inds)
        X_test = subset_selection(X_test, subset_inds)
    # SNV
    if do_snv:
        X_train = snv(X_train)
        X_test = snv(X_test)
    # Normalize
    if do_normalize:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
    return X_train, X_test

def median_reference(x: np.ndarray):
    medians = np.median(x, axis=0)
    return x / medians

def normalize(x: np.ndarray):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return (x-means) / stds

def moving_average(x: np.ndarray, w: int):
    kernel = np.ones(w)
    convolved = np.apply_along_axis(lambda i: np.convolve(i, kernel, mode='same'),
            axis=1, arr=x)
    return convolved / w

def snv(x: np.ndarray):
    assert x.ndim == 2, 'x must be a 2D array.'
    means = np.mean(x, axis=1)
    stds = np.std(x, axis=1)
    return (x - means[:, np.newaxis]) / stds[:, np.newaxis]

def subset_selection(x: np.ndarray, indices: np.ndarray):
    n_features = x.shape[1]
    mask = np.logical_and(indices < n_features, indices >= 0)
    indices = indices[mask]
    indices = np.unique(indices)
    indices = np.sort(indices)
    return np.take(x, indices, axis=1)

def create_table(x: np.ndarray) -> np.ndarray:
    ''' Reshapes an ND array to 2D.
    arg x: ND array
    return; 2D array '''
    if x.ndim <= 2:
        return x.reshape(-1)
    return x.reshape(-1, x.shape[-1])

def create_test_set(X: np.ndarray, Y: np.ndarray, test_frac: float) -> Tuple[np.ndarray]:
    assert X.ndim == 2, "X must be a 2D array."
    assert Y.ndim == 1, "Y must be a 1D array."
    assert test_frac < 1 or tes_frac > 0, "Invalid test set fraction."
    classes = np.unique(Y)
    inds = np.arange(X.shape[0], dtype=int)
    test_inds = []
    for c in classes:
        c_inds = np.where(Y==c)[0]
        n = int(c_inds.shape[0])
        n_test = int(np.floor(n*test_frac))
        test_inds += (np.random.choice(c_inds, n_test, replace=False)).tolist()
    test_inds = np.array(test_inds, dtype=int)
    train_inds = np.setdiff1d(inds, test_inds)
    X_test = np.take(X, test_inds, axis=0)
    Y_test = np.take(Y, test_inds, axis=0)
    X_train = np.take(X, train_inds, axis=0)
    Y_train = np.take(Y, train_inds, axis=0)
    return X_train, Y_train, X_test, Y_test

def one_hot_encode(labels: np.ndarray) -> Tuple[Dict, np.ndarray]:
    ''' One hot encodes labels. 
    arg labels: WxHxP np.array of uints
    return: tuple of dict and np.ndarray '''
    n_samples = len(labels)
    classes = np.unique(labels)
    n_classes = len(classes)
    if n_classes < 256:
        data_type = np.uint8
    else:
        data_type = np.uint16
    encoded_labels = np.zeros((n_samples, n_classes), dtype=data_type)
    encoded_labels[np.arange(n_samples), labels] = 1
    encoded_labels = encoded_labels.reshape((n_samples, n_classes))
    return encoded_labels

def create_tables(X: np.ndarray, Y: np.ndarray, test_frac):
    # Calibrate data and create tables
    X, Y = create_table(X), create_table(Y)
    # Create training and test set
    X_train, Y_train, X_test, Y_test = create_test_set(X, Y, frac=test_frac)
    return X_train, Y_train, X_test, Y_test

def remove_class(X: np.ndarray, Y: np.ndarray, label: int):
    inds = np.where(Y!=label)[0]
    to_remove = np.where(Y==label)[0]
    return np.take(X, inds, axis=0), np.take(Y, inds, axis=0)

def resample_dataset(X: np.ndarray, Y: np.ndarray, scale: float):
    classes, counts = np.unique(Y, return_counts=True)
    threshold = int(np.ceil(scale*np.min(counts)))
    inds = []
    for c, count in zip(classes, counts):
        c_inds = np.where(Y==c)[0]
        if count > threshold:
            count = threshold
        inds += (np.random.choice(c_inds, count, replace=False)).tolist()
    inds = np.array(inds, dtype=int)
    X_resampled = np.take(X, inds, axis=0)
    Y_resampled = np.take(Y, inds, axis=0)
    return X_resampled, Y_resampled

def hotellings_t2(X: np.ndarray, Y: np.ndarray, alpha: float, return_outliers: bool, show: bool,
        fig_num: int, fig_size: Tuple[int, int]):
    # PCA 
    model = decomposition.PCA(n_components=2)
    scores = model.fit_transform(X)
    angles = np.arctan2(scores[:,1], scores[:,0])
    radii = np.linalg.norm(scores, axis=1)
    # Make circles
    thetas = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
    circle = np.array((np.cos(thetas), np.sin(thetas)))
    # Compute 1-alpha ellipse
    sigmas = np.cov(np.array((scores[:,0], scores[:,1])))
    ci = np.sqrt(scipy.stats.chi2.ppf(1-alpha, 2))
    ellipse = np.transpose(circle).dot(np.linalg.cholesky(sigmas)*ci)
    a, b = np.max(ellipse[:,0]), np.max(ellipse[:,1])
    t = np.linspace(0, 2*np.pi, 100)
    # Calculate outliers
    indices = np.arange(X.shape[0])
    ss = np.sin(angles)
    cs = np.cos(angles)
    ellipse_radii = a*b/np.sqrt(a*a*ss*ss + b*b*cs*cs)
    outliers = np.squeeze(np.argwhere(radii > ellipse_radii))
    inliers = np.setdiff1d(indices, outliers)
    # Get scores
    outlier_scores = np.take(scores, outliers, axis=0)
    inlier_scores = np.take(scores, inliers, axis=0)
    if show:
        plt.figure(num=fig_num, figsize=fig_size)
        outlier_plot = plt.scatter(outlier_scores[:,0], outlier_scores[:,1], s=3, c='Red')
        inlier_plot = plt.scatter(inlier_scores[:,0], inlier_scores[:,1], s=3, c='Blue')
        ci_plot, = plt.plot(a*np.cos(t), b*np.sin(t), color='black')
        plt.grid(color='lightgray', linestyle='--')
        plt.title('PCA Scores and Hotelling\'s Conf. Bounds')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend([outlier_plot, inlier_plot, ci_plot], 
                ['Scores (outliers)', 'Scores', 
                    'Hotelling\'s {:.0f}%-CI bounds'.format((1-alpha)*100)])
        plt.show()

    if return_outliers:
        return outliers
    else:
        return np.array([])

def inspect_outliers(W: np.ndarray, X: np.ndarray, outliers: np.ndarray):
    indices = np.arange(X.shape[0])
    inliers = np.setdiff1d(indices, outliers)
    Xout = np.take(X, outliers, axis=0)
    Xin = np.take(X, inliers, axis=0)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.plot(W, Xin.T, c='blue', label='Samples')
    ax.plot(W, Xout.T, c='red', label='Samples (outliers)')
    ax.set_title('Preprocessed Samples')
    ax.set_ylabel('')
    ax.set_xlabel('Wave lengths [nm]')
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in new_labels:
            new_handles.append(handle)
            new_labels.append(label)
    plt.legend(new_handles, new_labels)
    plt.show()

def remove_outliers(X: np.ndarray, Y: np.ndarray, outliers: np.ndarray):
    indices = np.arange(X.shape[0])
    inliers = np.setdiff1d(indices, outliers)
    return np.take(X, inliers, axis=0), np.take(Y, inliers, axis=0)
