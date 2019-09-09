from abc import ABC, abstractmethod


class Model(ABC):
    ''' Abstract model class defining the structure of an ML model to be used
    with the MalCADe evaluation framework'''

    def __init__(self, *args, **kwargs):
        """Perform initial setup and handle arguments to the model"""
        pass

    @abstractmethod
    def train(self, inputs, outputs):
        """Perform training of the model"""
        pass

    @abstractmethod
    def run(self, input):
        """Run the model with new input data"""
        pass
