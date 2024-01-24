from src.jobs.load_data import load_dataset
from src.jobs.train import initialize_model, train
from src.jobs.evaluate import evaluate
from src.jobs.save import save_model

from omegaconf import OmegaConf
import os



cfgfile = 'cfg/simulation_config.yml'
cfg = OmegaConf.load(cfgfile)

x_train, y_train, x_test, y_test = load_dataset(cfg.data)
model = initialize_model(cfg)
train(model, x_train, y_train)
eval_score = evaluate(model, x_test, y_test)
save_dir = os.path.join('model_weight', cfg.model.name, cfg.model.version)
model_weight_path = save_model(model, save_dir = save_dir)
# log artifacts

    