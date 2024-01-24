from pydantic import BaseModel

class BaseConfig(BaseModel) :
    task_name: str
    model: str
    train_dataset: str
    validation_dataset: str
    batch_size: int
    evaluation_method: str

