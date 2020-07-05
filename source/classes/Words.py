from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .Characters import ASJPChar

from .Characters import Char

class Word:
    def __init__(self, chars: str):
        self.chars = self.str_to_chars(chars)

    @abstractmethod
    def get_feature_array(self):
        pass

    @abstractmethod
    def str_to_chars(self, str: str):
        pass


class ASJPWord(Word):
    def __init__(self, chars: str):
        super(ASJPWord, self).__init__(chars)

    def get_feature_array(self):
        return np.array([char.get_feature_vector() for char in self.chars])

    def str_to_chars(self, str: str):
        return [ASJPChar(char) for char in str]
               
    def __str__(self):
        return [char.get_char() for char in self.chars]


