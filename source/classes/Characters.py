from abc import ABC, abstractmethod
from typing import List


class Char(ABC):
    def __init__(self, char: str, features: List[str], vector: List[float]):
        self._char = char
        self._features = features
        self._vector = vector

    @abstractmethod
    def get_feature_val(self, feature: str):
        pass

    @abstractmethod
    def get_feature_vector(self):
        pass

    @abstractmethod
    def get_char(self):
        pass


class ASJPChar(Char):

    def __init__(self, asjp_char: str, features: List[str], vector: List[float]):
        super(ASJPChar, self).__init__(asjp_char, features, vector)
        
    def get_feature_val(self, feature: str):
        assert feature in self._features, "Feature {} not in dimensions!"
        return self._vector[self._features.index(feature)]
        
    def get_feature_vector(self):
        return self._vector

    def get_char(self):
        return self._char
        
    def __str__(self):
        str = self._char + " ("
        for i, feature in enumerate(self._features):
            if self._vector[i] == 1:
                str += " " + self._features[i]
        str += " )"
        return str
