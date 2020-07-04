##################################################################################################
### I copied the consonant features from the CONLING paper by Taraka (2016)
##################################################################################################

from abc import ABC, abstractmethod
import numpy as np
from typing import List

from .Vectorizers import Vectorizer, ASJPVectorizer

asjp_cons_dims = [
    'voiced', 'labial', 'dental', 'alveolar', 'palatal/post-alveolar', 'velar', 'uvular',
    'glottal', 'stop', 'fricative', 'affricate', 'nasal', 'click', 'approximant',
    'lateral', 'rhotic'
]

# TODO: do this for vowels and join

class Char(ABC):
    def __init__(self, char: str, dims: List[str], vectorizer: Vectorizer):
        self.char = char
        self.dims = dims
        self.vector= vectorizer.apply(char)

    @abstractmethod
    def get_feature_val(self, feature: str):
        assert feature in self.dims, "Feature {} not in dimensions!"
        return self.vector[self.dims.index(feature)]

    @abstractmethod
    def get_feature_vector(self):
        pass


class ASJPChar(Char):

    def __init__(self, asjp_char: str):
        super(ASJPChar, self).__init__(asjp_char, asjp_cons_dims, ASJPVectorizer())
        
    def get_feature_val(self, feature: str):
        super(ASJPChar, self).get_feature_val(feature)
        
    def get_feature_vector(self):
        return self.vector
        
    def __str__(self):
        str = self.char + " ("
        for i, feature in enumerate(self.dims):
            if self.vector[i] == 1:
                str += " " + self.dims[i]
        str += " )"
        return str
