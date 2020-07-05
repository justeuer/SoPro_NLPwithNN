from abc import ABC, abstractmethod
from .Characters import ASJPChar
from .Words import ASJPWord

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
        return ASJPWord(word)
