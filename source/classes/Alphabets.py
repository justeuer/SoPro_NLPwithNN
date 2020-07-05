from abc import ABC, abstractmethod
from .Characters import ASJPChar
from Constants import asjp_char_dims, asjp_char_as_vector
from typing import Dict, List

class Alphabet(ABC):
    def __init__(self, name: str):
        print("Created {} alphabet Object".format(name))

    @abstractmethod
    def translate(self, word: str):
        pass

class ASJPAlphabet(Alphabet):
    def __init__(self):
        super(ASJPAlphabet, self).__init__("ASJP")

    def translate(self, word: str):
        chars = []
        for char in word:
            chars.append(ASJPChar(char))
        return chars
