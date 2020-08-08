from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Dict, List

from Characters import ASJPChar
from Words import ASJPWord


class Alphabet(ABC):
    def __init__(self, header_row=0, chars_col=0):
        self.header_row = header_row
        self.chars_col = chars_col
        # self._load(csv)

    @abstractmethod
    def translate_and_align(self, cognates: Dict[str, str]):
        """
        Translates and aligns words in a cognate set into a classes.Words.Word object
        Parameters
        ----------
        cognates
            A dictionary containing a cognate set, keys languages, values words
        Returns
            An Dict of [str, classes.Words.Word]
        -------

        """
        pass

    @abstractmethod
    def _create_char(self, char: str):
        """
        Creates a classes.Characters.Character object from a string
        Parameters
        ----------
        char
            The character from which the ASJP char should be translated
        Returns
            An ASJPChar object
        -------
        """
        pass

    @ abstractmethod
    def _find_chars(self, word: str):
        """
        Creates a character representation of a word
        Parameters
        ----------
        word
            The word to be transformed
        Returns
            The character representation of the word ready for alignment
        -------

        """
        pass

    @abstractmethod
    def _load(self, path: Path):
        """
        Loads the feature set for a classes.Alphabets.Alphabet from a csv file
        Parameters
        ----------
        path
            The path to the csv file
        -------
        """
        pass


class ASJPAlphabet(Alphabet):

    def __init__(self,
                 csv: Path,
                 start_symbol="<start>",
                 stop_symbol="<stop>",
                 pad_symbol="<pad>",
                 header_row=0,
                 chars_col=0):
        super(ASJPAlphabet, self).__init__(header_row, chars_col)
        self._features = []
        self._alphabet = []
        self._dict = {}
        self.start_symbol = start_symbol
        self.stop_symbol = stop_symbol
        self.pad_symbo = pad_symbol
        self._load(csv)

    def translate_and_align(self, cognates: Dict[str, str]):
        #TODO: align!

        to_align = {}
        for lang, word in cognates.items():
            to_align[lang] = self._find_chars(word)

        max_l = max([len(vals) for vals in to_align.values()])
        for lang, chars in to_align.items():
            # pad
            if len(chars) < max_l:
                for _ in range(max_l - len(chars)):
                    chars.append("-")
            # append start/stop symbol
            chars.insert(0, self.start_symbol)
            chars.append(self.stop_symbol)
            # yield
            print(chars)
            yield lang, ASJPWord([self._create_char(c) for c in chars])

    def _load(self, path: Path):
        rows = path.open().read().split("\n")
        self._features = rows[self.header_row].split(",")[1:]
        for row in rows[self.header_row + 1:]:
            if row != "":
                cols = row.split(",")
                assert len(cols)-1 == len(self._features), \
                    "Not enough features found, expected {}, got {}".format(len(self._features), len(cols))
                char = cols[self.chars_col]
                self._alphabet.append(char)
                vec = []
                for feature_val in cols[self.header_row + 1:]:
                    vec.append(int(feature_val))
                self._dict[char] = vec
        print(self._alphabet)

    def _find_chars(self, word: str):

        # not the nicest implementation
        def find_groups(chunk: str, chars_purged: List[str]):
            if "(" in chunk:
                split = chunk.split("(")
                s0 = split[0]
                s1 = re.search(r'(.*?)\)', split[1]).group(1)
                s2 = split[1].split(")")[1]
                chars_purged.append(find_groups(s0, chars_purged))
                chars_purged.append(find_groups(s1, chars_purged))
                chars_purged.append(find_groups(s2, chars_purged))
            elif "[" in chunk:
                split = chunk.split("[")
                s0 = split[0]
                s1 = re.search(r'(.*?)\]', split[1]).group(1)
                s1 = "-" * len(s1)
                s2 = split[1].split("]")[1]
                chars_purged.append(find_groups(s0, chars_purged))
                chars_purged.append(find_groups(s1, chars_purged))
                chars_purged.append(find_groups(s2, chars_purged))
            else:
                for c in chunk.split():
                    chars_purged.append(c)

            chars_ = []
            for c in chars_purged:
                if isinstance(c, str):
                    chars_.append(c)
            return chars_

        chars_ = ""
        for group in find_groups(word, []):
            for char in group:
                chars_ += char

        chars = []
        i = 0
        while i <= len(chars_) - 1:
            c = chars_[i]
            if i < len(chars_) - 1:
                c_next = chars_[i + 1]
                if c_next == ":":
                    c += c_next
                    i += 2
                    chars.append(c)
                    continue
            chars.append(c)
            i += 1
        return chars

    def _create_char(self, char: str):
        assert char in self._alphabet, "Unknown character {}".format(char)
        return ASJPChar(char, self._features, self._dict[char])

    def __str__(self):
        s = "*** ASJP alphabet class ***\n"
        for ci, (char, vector) in enumerate(self._dict.items()):
            feature_vals = "{}\t{}\t(".format(ci, char)
            for vi, val in enumerate(vector):
                if int(val) == 1:
                    feature_vals += " {} ".format(self._features[vi + 1])
            feature_vals += ")"
            s += feature_vals + "\n"
        return s


if __name__ == '__main__':
    csv = Path("../../data/alphabets/asjp.csv")
    asjp = ASJPAlphabet(csv)
    cognates = {
        'lat': "ego:",
        'it': "i:-o",
        'spa': "S-o",
        'fr': "Z-3",
        'rom': "y-(ew)"
    }
    for lang, asjp_word in asjp.translate_and_align(cognates):
        print(asjp_word.get_feature_array())
