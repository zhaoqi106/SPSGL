import torch
from omegaconf import DictConfig, open_dict
import pandas as pd
import re
from joblib import dump, load
import os
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from torch.utils.backcompat import keepdim_warning


def normalize(data):
    data = data.float()
    num_subject, num_roi, num_features = data.shape
    data_flat = data.view(num_subject, -1)
    min = data_flat.min(dim=1, keepdim=True)[0]
    max = data_flat.max(dim=1, keepdim=True)[0]
    normalized_flat = (data_flat - min) / (max - min + 1e-8)
    normalized = normalized_flat.view(num_subject, num_roi, num_features)
    return  normalized

def standardize_per_subject(data):
    data = data.float()
    num_subject, num_roi, num_features = data.shape
    data_flat = data.view(num_subject, -1)  # (num_subject, num_roi * num_features)
    mean = data_flat.mean(dim=1, keepdim=True)  # (num_subject, 1)
    std = data_flat.std(dim=1, keepdim=True)  # (num_subject, 1)
    standardized_flat = (data_flat - mean) / (std + 1e-8)
    standardized = standardized_flat.view(num_subject, num_roi, num_features)
    return standardized

def normalize(data):
    num_subject, num_roi, num_features = data.shape
    data_flat = data.view(num_subject, -1)
    min = data_flat.min(dim=1, keepdim=True)[0]
    max = data_flat.max(dim=1, keepdim=True)[0]
    normalized_flat = (data_flat - min) / (max - min + 1e-8)
    normalized = normalized_flat.view(num_subject, num_roi, num_features)
    return  normalized

def create_folds(fc_array, node_array, labels_array, group=None, n_splits=5, random_state=42):

    folds = []
    if group is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(fc_array, labels_array)
    else:
        group = np.asarray(group)
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(fc_array, labels_array, groups=group)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train_fc, X_test_fc = fc_array[train_idx], fc_array[test_idx]
        X_train_node, X_test_node = node_array[train_idx], node_array[test_idx]
        y_train, y_test = labels_array[train_idx], labels_array[test_idx]

        folds.append({
            "fold": fold_idx,
            "train_fc_data": torch.from_numpy(X_train_fc).float(),
            "train_node_data": normalize(torch.from_numpy(X_train_node).float()),
            "train_labels": torch.from_numpy(y_train),
            "test_fc_data": torch.from_numpy(X_test_fc).float(),
            "test_node_data": normalize(torch.from_numpy(X_test_node).float()),
            "test_labels": torch.from_numpy(y_test)
        })

    return folds

def load_data(cfg: DictConfig):
    data = load(cfg.dataset.path)
    label_threshold = cfg.dataset.label_threshold
    fc_array = data["fc"]
    node_array = data["time_series_freq"]
    labels_ori_array = data[cfg.dataset.label_columns]
    valid_indices = ((labels_ori_array >= label_threshold[1]) | (labels_ori_array <= label_threshold[0])) & (labels_ori_array >= 0)
    fc_array = fc_array[valid_indices]  # 过滤 npy 数据
    node_array = node_array[valid_indices]
    labels_ori_array = labels_ori_array[valid_indices]
    labels_array = np.where(labels_ori_array >= label_threshold[1], cfg.dataset.label_sign[1], cfg.dataset.label_sign[0])
    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = fc_array.shape[1:]
    group = data["group"] if "group" in data.keys() else None
    folds = create_folds(fc_array,node_array, labels_array, group)
    return folds

