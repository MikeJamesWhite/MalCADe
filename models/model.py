from abc import ABC, abstractmethod


class Model(ABC):
    ''' Abstract model class defining the structure of an ML model to be used
    with the MalCADe evaluation framework'''

    # Perform any initial setup and handle arguments to the model
    def __init__(self, *args, **kwargs):
        pass

    # Perform any required training of the model
    @abstractmethod
    def train(self, inputs, outputs):
        pass

    # Run the model with a new input
    @abstractmethod
    def run(self, input):
        pass
