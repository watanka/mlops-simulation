from src.jobs.load_data import load_dataset
from src.jobs.train import initialize_model, train
from src.jobs.evaluate import evaluate
from src.jobs.save import save_model

import mlflow
from datetime import datetime
from omegaconf import OmegaConf
import os



cfgfile = 'cfg/simulation_config.yml'
cfg = OmegaConf.load(cfgfile)

task_name=cfg.task_name
experiment_name=task_name
now=datetime.now().strftime("%Y%m%d_%H%M%S")
run_name=f'{task_name}_{now}'
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))

with mlflow.start_run() :
    x_train, y_train, x_test, y_test = load_dataset(cfg.data)
    model = initialize_model(cfg)
    # mlflow.log_artifact(cfg, 'cfg')

    train(model, x_train, y_train)
    

    eval_score = evaluate(model, x_test, y_test)
    mlflow.log_metric('eval score', eval_score)
    save_dir = os.path.join('model_weight', cfg.model.name, cfg.model.version)
    model_weight_path = save_model(model, save_dir = save_dir)
    # log artifacts

    