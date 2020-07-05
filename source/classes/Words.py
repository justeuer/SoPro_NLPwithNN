from abc import ABC, abstractmethod
import numpy as np

from Constants import START_SYMBOL, STOP_SYMBOL

from .Characters import ASJPChar

from .Characters import Char

class Word:
    def __init__(self, chars: str):
        self.chars = self._str_to_chars(START_SYMBOL + chars + STOP_SYMBOL)

    @abstractmethod
    def get_feature_array(self):
        pass

    @abstractmethod
    def _str_to_chars(self, str: str):
        pass


class ASJPWord(Word):
    def __init__(self, chars: str):
        super(ASJPWord, self).__init__(chars)

    def get_feature_array(self):
        return np.array([char.get_feature_vector() for char in self.chars])

    def _str_to_chars(self, chars: str):
        #print(chars)
        return [ASJPChar(char) for char in chars]
               
    def __str__(self):
        str = ""
        for char in self.chars:
            str += char.get_char()
        return str


