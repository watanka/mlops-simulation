from model import AbstractModel
import config
from build import build_dataset, build_model

import os
from dotenv import load_dotenv
from datetime import datetime
import mlflow

load_dotenv()

def train(cfg: config.BaseConfig):
    
    task_name=cfg.task_name
    experiment_name=task_name
    now=datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name=f'{task_name}_{now}'
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
    print(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(experiment_name=experiment_name)

    
    with mlflow.start_run() :
        mlflow.log_params(cfg.model_dump_json())

        build
        

if __name__ == '__main__':
    cfg = config.BaseConfig(
        task_name='sample-experiment',
        model='SimulationModel',
        train_dataset='sample-trainset',
        validation_dataset='sample-valset',
        batch_size=16,
        evaluation_method='f1-score'
    )


    train(cfg)

    with mlflow.start_run():
        mlflow.log_param('lr', 0.01)
    