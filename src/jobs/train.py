from omegaconf import DictConfig
from src.models.abstract_model import AbstractModel
from src.models.concrete_model import ConcreteModel

def initialize_model(cfg: DictConfig) -> AbstractModel:
    model = ConcreteModel(cfg)
    model.define_base_model()

    return model

def train(model: AbstractModel, x_train, y_train):
    model.train(x_train, y_train)
