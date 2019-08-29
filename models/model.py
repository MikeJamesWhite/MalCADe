from abc import ABC, abstractmethod


class Model(ABC):
    ''' Abstract model class defining the structure of a ML model to be used
    with the MalSeg testing framework'''

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

if __name__ == '__main__':
    print("Hello from model.py")
