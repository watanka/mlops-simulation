from src.models.abstract_model import AbstractModel

import os

def save_model(model:AbstractModel, save_dir: str):
    weight_path = os.path.join(save_dir, 'model_weight.pt')
    os.makedirs(save_dir, exist_ok=True)
    model.save(weight_path)

    return weight_path


def save_model_as_onnx(model:AbstractModel, save_dir: str):
    raise NotImplementedError

