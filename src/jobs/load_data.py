from omegaconf import DictConfig
import os

def load_dataset(data_cfg: DictConfig):

    train_x = os.path.join(data_cfg.train, 'x')
    train_y = os.path.join(data_cfg.train, 'y')

    test_x = os.path.join(data_cfg.test, 'x')
    test_y = os.path.join(data_cfg.test, 'y')


    return train_x, train_y, test_x, test_y