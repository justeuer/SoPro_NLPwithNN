from abc import ABC, abstractmethod
import numpy as np
from typing import List

from Characters import ASJPChar, Char


class Word:
    def __init__(self, chars: List[Char]):
        self.chars = chars

    def get_feature_array(self):
        return np.array([char.get_feature_vector() for char in self.chars])

    def get_chars(self):
        return self.chars


class ASJPWord(Word):
    def __init__(self, chars: List[ASJPChar]):
        super(ASJPWord, self).__init__(chars)

    def __str__(self):
        str = ""
        for char in self.chars:
            str += char.get_char()
        return str


