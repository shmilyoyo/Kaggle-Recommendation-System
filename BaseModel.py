from abc import ABCMeta, abstractmethod

class BaseModel(metaclass=ABCMeta):

    def __init__(self, model_id):
        self.model_id = model_id
    
    @abstractmethod
    def runModel(self):
        pass