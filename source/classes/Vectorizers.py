##################################################################################################
### Just a Wrapper class
##################################################################################################

from abc import ABC, abstractmethod
import  numpy as np
from typing import Dict, List
from Constants import asjp_char_as_vector


class Vectorizer(ABC):
    def __init__(self, map: Dict[str, List[float]]):
        self.map = map

    @abstractmethod
    def apply(self, char: str):
        pass


class ASJPVectorizer(Vectorizer):
    def __init__(self):
        super(ASJPVectorizer, self).__init__(asjp_char_as_vector)

    def apply(self, char: str):
        assert char in self.map.keys(), "Char {} not in map!".format(char)
        return self.map[char]
