import wandb
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
import torch

def model_training(cfg: DictConfig,dataloader):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloader, logger)

    training.train()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dataloaders = dataset_factory(cfg)
    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"

    for dataloader in dataloaders:
        run = wandb.init(project=cfg.project, reinit=True,
                         group=f"{group_name}", tags=[f"{cfg.dataset.name}"])
        model_training(cfg, dataloader)
        run.finish()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    main()
