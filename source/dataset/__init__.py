from omegaconf import DictConfig
from .process_dataset import load_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['hcp', 'abide']
    dataloaders = []
    datasets = load_data(cfg)
    for split_dataset in datasets:
        dataloaders.append(init_dataloader(cfg, split_dataset))

    return dataloaders
