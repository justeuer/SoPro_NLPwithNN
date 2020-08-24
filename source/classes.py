from pathlib import Path
import numpy as np
from tensorflow import keras
import re
from typing import Dict, List


class Char(object):
    """ Class to represent either an IPA, ASJP or ordinary latin char"""
    def __init__(self,
                 char: str,
                 features: List[str],
                 vector: List[float]):
        self._features = features
        self._char = char
        self._vector = vector

    def get_feature_val(self, feature: str):
        assert feature in self._features, "Feature {} not in dimensions!"
        return self._vector[self._features.index(feature)]

    def get_feature_vector(self):
        return self._vector

    def get_char(self):
        return self._char

    def __str__(self):
        s = self._char + " ("
        for i, feature in enumerate(self._features):
            if self._vector[i] == 1:
                s += " " + self._features[i]
        s += " )"
        return s


class Word:
    """ Wrapper class, contains a list of chars """
    def __init__(self, chars: List[Char]):
        self.chars = chars

    def get_feature_array(self):
        """
        Agglutinates the feature vectors of the chars of the word
        Returns
            An numpy array of the size (n_chars x n_features)
        -------

        """
        return np.array([char.get_feature_vector() for char in self.chars])

    def get_chars(self):
        return self.chars

    def __str__(self):
        s = ""
        for char in self.chars:
            s += char.get_char()
        return s

    def __len__(self):
        return len(self.chars)


class Alphabet(object):
    """ Class that manages the translation of cognate sets into vector arrays """
    match_long_vowel = re.compile(r"^[aɑɐeəɨiɪoɔœɒuʊɛyʏø]ː")
    match_nasal_vowel = re.compile(r"^[aɐeiwoɨuɑɔɛ]̃") # not the best way to do it (nasal tilde not visible, use unicode instead?
    match_long_consonant = re.compile(r"^[pɸfbβvtθsdðzcɟçʝʃʒkxχgɣmnɲŋlɫʎrɾwɥ]ː")
    match_affricate = re.compile(r"^[tdɟ]͡[szʃʒʝ]")
    match_long_affricate = re.compile(r"^[tdɟ]͡[szʃʒʝ]ː")

    def __init__(self,
                 csv: Path,
                 encoding='utf-16',
                 start_symbol="<start>",
                 stop_symbol="<stop>",
                 pad_symbol="<pad>",
                 empty_symnol="-",
                 header_row=0,
                 chars_col=0):
        self._features = []
        self._alphabet = []
        self._dict = {}
        self.start_symbol = start_symbol
        self.stop_symbol = stop_symbol
        self.pad_symbol = pad_symbol
        self.empty_symbol = empty_symnol
        self.header_row = header_row
        self.chars_col = chars_col
        self.encoding = encoding
        self._load(csv, encoding=self.encoding)
        self._char_labels = self._create_char_labels()

    def _create_char_labels(self):
        """
        Creates a
        Returns
        -------

        """
        char_labels = {}
        char2index = {char: i for i, char in enumerate(list(self._dict.keys()))}

        for char, one_hot_vector in zip(char2index.keys(),
                                 keras.utils.to_categorical(list(char2index.values()), num_classes=len(char2index))):
            char_labels[char] = one_hot_vector
        return char_labels

    def translate_and_align(self, cognates: Dict[str, str]):
        """
        Translates and aligns words in a cognate set into a classes.Word object
        Parameters
        ----------
        cognates
            A dictionary containing a cognate set, keys languages, values words
        Returns
            An Dict of [str, classes.Words.Word]
        -------
        """
        pass

    def get_char_labels(self):
        return self._char_labels

    def translate(self, word: str):
        """
        Translates a single word into a classes.Word object
        Parameters
        ----------
        word
            The word to be translated
        Returns
            The Word object containing the list of classes.Char object
        -------

        """
        chars_ = []
        chars_.append(self.start_symbol)
        self._find_chars(word, chars_)
        chars_.append(self.stop_symbol)
        chars = [self._create_char(char_) for char_ in chars_]
        return Word(chars)

    def _find_chars(self, chunk: str, chars: List[str]):

        """
        Recursively parses an database entry, looking for '()' (phonological split) and '[]' (spurious morphemes).
        Parameters
        ----------
        chunk
            The rest of the word to parse
        chars
            A list of parsed substrings until now
        -------
        """
        # break condition
        if len(chunk) == 0:
            return
        else:
            if bool(self.match_long_affricate.match(chunk)):
                group = self.match_long_affricate.match(chunk).group(0)
            elif bool(self.match_affricate.match(chunk)):
                group = self.match_affricate.match(chunk).group(0)
            elif bool(self.match_nasal_vowel.match(chunk)):
                group = self.match_nasal_vowel.match(chunk).group(0)
            elif bool(self.match_long_consonant.match(chunk)):
                group = self.match_long_consonant.match(chunk).group(0)
            elif bool(self.match_long_vowel.match(chunk)):
                group = self.match_long_vowel.match(chunk).group(0)
            else:
                group = chunk[0]

            chunk_ = chunk[len(group):]
            chars.append(group)
            self._find_chars(chunk_, chars)

    def _load(self, path: Path, encoding='utf-16'):
        """
        Loads the feature set for a classes.Alphabets.Alphabet from a csv file
        Parameters
        ----------
        path
            The path to the csv file
        -------
        """
        rows = path.open(encoding=encoding).read().split("\n")
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

    def _create_char(self, char: str):
        """
        Creates a classes.Characters.Character object from a string
        Parameters
        ----------
        char
            The character from which the ASJP char should be translated
        Returns
            A Char object
        -------
        """
        assert char in self._alphabet, "Unknown character '{}'".format(char)
        return Char(char, self._features, self._dict[char])

    def __str__(self):
        s = "*** ASJP alphabet class ***\n"
        for ci, (char, vector) in enumerate(self._dict.items()):
            feature_vals = "{}\t{}\t(".format(ci, char)
            for vi, val in enumerate(vector):
                if int(val) == 1:
                    feature_vals += " {} ".format(self._features[vi])
            feature_vals += ")"
            s += feature_vals + "\n"
        return s

    def __len__(self):
        return len(self._dict)


class Asjp2Ipa(object):
    def __init__(self,
                 sca,
                 ignored_symbols: List[str],
                 start_symbol="<start>",
                 stop_symbol="<stop>",
                 pad_symbol="<pad>",
                 empty_symbol="-"):
        self.sca = sca
        self.empty_symbol = empty_symbol
        self.start_symbol = start_symbol
        self.stop_symbol = stop_symbol
        self.pad_symbol = pad_symbol
        self.ignored_symbols = ignored_symbols

    def convert(self, chars: List[Char]):
        s = ""
        for char in chars:
            char_ = char.get_char()
            if char_ == self.empty_symbol:
                s += self.empty_symbol
                continue
            # ignore start and stop symbols
            if char_ == self.start_symbol or char_ == self.stop_symbol:
                continue
            # lingpy doesn't convert nasal vowels
            if char.get_feature_val('nasal') == 1 and char.get_feature_val('stop') == 0:
                char_ = char_.replace("ɐ̃", "ɐ").\
                            replace("ɔ̃", "ɔ").\
                            replace("w̃", "w").\
                            replace("ɛ̃", "ɛ").replace("ɑ̃", "ɑ")
            # also the palatal affricate is not recognized
            if char.get_feature_val('palatal') == 1 and char.get_feature_val('affricate') == 1:
                char_ = char_.replace("ɟ͡ʝ", "d͡ʒ")
            # vowel length is ignored
            for ignored_symbol in self.ignored_symbols:
                char_ = char_.replace(ignored_symbol, "")
            # handle empty string
            if char_ == "":
                continue

            char_ = self.sca.converter[char_]
            s += char_

        return s


if __name__ == '__main__':
    ipa = Alphabet(Path("../data/alphabets/ipa.csv"), encoding='utf-16')
    print(ipa.get_char_labels())
    asjp = Alphabet(Path("../data/alphabets/asjp.csv"), encoding='ascii')
    print(asjp.get_char_labels())
