import abc

class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def define_base_model(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError

    