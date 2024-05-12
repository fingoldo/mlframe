
from abc import ABC, abstractmethod, abstractproperty


class BusinessModel(ABC):
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        pass

    def prepare_features(self, *args, **kwargs):
        pass        