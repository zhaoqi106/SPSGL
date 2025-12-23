from omegaconf import DictConfig
from .SPSGL import SPSGL


def model_factory(config: DictConfig):
    return eval(config.model.name)(config).cuda()
