from src.models.abstract_model import AbstractModel
from src.models.concrete_model import ConcreteModel
import numpy as np

def evaluate(model: AbstractModel, x_test, y_test):
    return model.evaluate(x_test, y_test)

