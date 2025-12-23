from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data,add_gaussian_noise,drop_edge
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
from torch_geometric.data import Data, Batch

def build_graphs(fc_data, node_data):
    """
    从功能连接矩阵和节点特征构建批量图。

    Args:
        fc_data: Tensor of shape (batch_size, num_roi, num_roi)
        node_data: Tensor of shape (batch_size, num_roi, num_features)

    Returns:
        batch_graph: Batch object containing x, edge_index, edge_attr, batch
    """
    batch_size, num_roi, _ = fc_data.shape
    graph_list = []

    i, j = torch.triu_indices(num_roi, num_roi, offset=1, device=fc_data.device)

    for b in range(batch_size):
        fc_values = fc_data[b, i, j]
        edge_index = torch.stack([i, j], dim=0)
        edge_attr = fc_values.unsqueeze(1)
        graph = Data(x=node_data[b], edge_index=edge_index, edge_attr=edge_attr)
        graph_list.append(graph)

    batch_graph = Batch.from_data_list(graph_list).to(fc_data.device)
    return batch_graph

class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.test_dataloader, pos_weight = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        # self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda(), reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for fc_data, node_data, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            fc_data, node_data, label = fc_data.cuda(),node_data.cuda(), label.cuda()
            # fc_data = add_gaussian_noise(fc_data, std=0.05)
            # fc_data = drop_edge(fc_data, drop_prob=0.1)
            if self.config.preprocess.continus:
                fc_data, node_data, label = continus_mixup_data(
                    fc_data, node_data, y=label)
            batch_graph = build_graphs(fc_data, node_data)
            # predict = self.model(fc_data,node_data)
            predict, edge_attr = self.model(batch_graph)

            # loss = self.loss_fn(predict, label)
            # ✅ 如果是 BCEWithLogitsLoss：直接传入 logits 和 float 标签
            loss = self.loss_fn(predict[:, 1], label[:, 1])+ 0.1 * F.mse_loss(edge_attr, fc_data)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            # wandb.log({"LR": lr_scheduler.lr,
            #            "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for fc_data, node_data, label in dataloader:
            fc_data, node_data, label = fc_data.cuda(),node_data.cuda(), label.cuda()
            batch_graph = build_graphs(fc_data, node_data)
            output, edge_attr = self.model(batch_graph)

            label = label.float()

            loss = self.loss_fn(output, label)+ 0.1 * F.mse_loss(edge_attr, fc_data)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def generate_save_learnable_matrix(self):

        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        best_test_auc = 0
        best_model_path = None
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)


            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'Test Spe:{test_result[-2]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val Loss": self.val_loss.avg,
            })

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)