import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    dataset = utils.TensorDataset(
        final_timeseires[:train_length+val_length+test_length],
        final_pearson[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length])
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
from collections import Counter


def calculate_label_statistics(dataloader: utils.DataLoader, device='cuda', label_position=2):
    """
    统计每个类标签在 DataLoader 中的数量和比例（兼容多输入数据结构）

    :param dataloader: PyTorch DataLoader 对象
    :param device: 使用的设备，默认为 'cuda'
    :param label_position: 标签在 batch 中的位置索引（默认2，即第3个元素）
    :return: 标签数量、标签比例（字典形式）
    """
    label_counts = Counter()
    total_samples = 0

    for batch in dataloader:
        # 动态获取标签（根据位置或属性）
        if isinstance(batch, (list, tuple)):
            labels = batch[label_position]  # 按位置索引获取
        elif hasattr(batch, 'y'):
            labels = batch.y  # 支持属性访问方式
        else:
            raise ValueError("无法确定标签位置，请检查batch数据结构")

        # 统一标签处理逻辑
        labels = labels.to(device)
        if labels.ndim > 1:  # 处理one-hot编码
            label_indices = labels.argmax(dim=1)
        else:  # 直接是类别索引
            label_indices = labels.long()

        # 更新统计
        label_counts.update(label_indices.cpu().numpy().flatten().tolist())
        total_samples += len(labels)

    # 计算比例
    label_percentages = {k: v / total_samples for k, v in label_counts.items()}

    return label_counts, label_percentages

def init_dataloader(cfg: DictConfig,
                    split_dataset: list) -> List[utils.DataLoader]:
    # train_data = [torch.from_numpy(train_fc).float(), torch.from_numpy(train_text).float(), torch.from_numpy(train_labels).float(), train_ids]
    # val_data = [torch.from_numpy(val_fc).float(), torch.from_numpy(val_text).float(), torch.from_numpy(val_labels).float(), val_ids]
    # train_fc, train_text, train_labels, _ = train_data
    # val_fc, val_text, val_labels, _ = val_data
    train_fc, train_node, train_labels = split_dataset["train_fc_data"], split_dataset["train_node_data"], split_dataset["train_labels"]
    test_fc, test_node, test_labels = split_dataset["test_fc_data"], split_dataset["test_node_data"], split_dataset["test_labels"]

    train_length = train_fc.shape[0]
    test_length = test_fc.shape[0]
    train_labels = F.one_hot(train_labels.to(torch.int64))
    test_labels = F.one_hot(test_labels.to(torch.int64))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # split = StratifiedShuffleSplit(
    #     n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
    # for train_index, test_valid_index in split.split(final_timeseires, stratified):
    #     final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
    #         train_index], final_pearson[train_index], labels[train_index]
    #     final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
    #         test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
    #     stratified = stratified[test_valid_index]
    #
    # split2 = StratifiedShuffleSplit(
    #     n_splits=1, test_size=test_length)
    # for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
    #     final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
    #         test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
    #     final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
    #         valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    # final_timeseires_train, final_timeseires_val, final_timeseires_test = tabfpn_embedding(
    #     [final_timeseires_train,labels_train],[final_timeseires_val, labels_val],[final_timeseires_test,labels_test])

    train_dataset = utils.TensorDataset(
        train_fc,
        train_node,
        train_labels
    )

    # val_dataset = utils.TensorDataset(
    #     test_fc,
    #     test_node,
    #     test_labels
    # )
    test_dataset = utils.TensorDataset(
        test_fc,
        test_node,
        test_labels
    )

    # train_labels = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))]).argmax(dim=1)
    # class_counts = torch.bincount(train_labels)
    # class_weights = 1.0 / class_counts.float()
    # sample_weights = class_weights[train_labels]
    # train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    #
    # train_dataloader = utils.DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, sampler=train_sampler,
    #                                     drop_last=cfg.dataset.drop_last)
    # val_dataloader = utils.DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)
    # test_dataloader = utils.DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)


    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    # val_dataloader = utils.DataLoader(
    #     val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    device = cfg.get("device", "cuda")
    train_label_counts, train_label_percentages = calculate_label_statistics(train_dataloader, device=device)
    # val_label_counts, val_label_percentages = calculate_label_statistics(val_dataloader, device=device)
    test_label_counts, test_label_percentages = calculate_label_statistics(test_dataloader, device=device)


    # 打印统计信息
    print("Train Label Counts:", train_label_counts)
    print("Train Label Percentages:", train_label_percentages)

    # print("Validation Label Counts:", val_label_counts)
    # print("Validation Label Percentages:", val_label_percentages)

    print("Test Label Counts:", test_label_counts)
    print("Test Label Percentages:", test_label_percentages)

    # return [train_dataloader, val_dataloader, test_dataloader, torch.tensor([train_label_counts[0] / train_label_counts[1]])]
    return [train_dataloader,  test_dataloader, torch.tensor([train_label_counts[0] / train_label_counts[1]])]

def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
    for train_index, test_valid_index in split.split(final_timeseires, stratified):
        final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
            test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
        stratified = stratified[test_valid_index]

    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_length)
    for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
        final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
            test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
        final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
            valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    # final_timeseires_train, final_timeseires_val, final_timeseires_test = tabfpn_embedding(
    #     [final_timeseires_train,labels_train],[final_timeseires_val, labels_val],[final_timeseires_test,labels_test])

    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    # train_labels = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))]).argmax(dim=1)
    # class_counts = torch.bincount(train_labels)
    # class_weights = 1.0 / class_counts.float()
    # sample_weights = class_weights[train_labels]
    # train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    #
    # train_dataloader = utils.DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, sampler=train_sampler,
    #                                     drop_last=cfg.dataset.drop_last)
    # val_dataloader = utils.DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)
    # test_dataloader = utils.DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)


    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    device = cfg.get("device", "cuda")
    train_label_counts, train_label_percentages = calculate_label_statistics(train_dataloader, device=device)
    val_label_counts, val_label_percentages = calculate_label_statistics(val_dataloader, device=device)
    test_label_counts, test_label_percentages = calculate_label_statistics(test_dataloader, device=device)


    # 打印统计信息
    print("Train Label Counts:", train_label_counts)
    print("Train Label Percentages:", train_label_percentages)

    print("Validation Label Counts:", val_label_counts)
    print("Validation Label Percentages:", val_label_percentages)

    print("Test Label Counts:", test_label_counts)
    print("Test Label Percentages:", test_label_percentages)

    return [train_dataloader, val_dataloader, test_dataloader]