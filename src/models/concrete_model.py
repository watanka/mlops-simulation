from src.models.abstract_model import AbstractModel

from omegaconf import DictConfig

class ConcreteModel(AbstractModel):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def train(self, x_train, y_train):
        print('start training!')
        
    def predict(self, x):
        print('start prediction')
        return self.model.predict(x)

    def define_base_model(self):
        '''Abstract Pytorch Model'''
        class SampleModel:
            def predict(self, x):
                return f'result on {x}'
            
            def __repr__(self) -> str:
                return __class__.__name__
        
        self.model = SampleModel()

    def save(self, save_dir: str) :
        with open(save_dir, 'w') as f :
            f.write('sample model')
            
        print(f'save the model {self.model} in {save_dir}')

    def evaluate(self, test_x, test_y):
        print(f'evaluate using {test_x} and {test_y}.')
        return 100

