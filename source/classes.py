import pandas as pd
from pathlib import Path
import numpy as np
from tensorflow import keras
import re
from typing import Dict, List
from collections import Counter
from matplotlib import pyplot as plt
import nltk

from utils import cos_sim


class Char(object):
    """ Class to represent either an IPA, ASJP or ordinary latin char"""

    def __init__(self,
                 char: str,
                 features: List[str],
                 vector: List[float]):
        self.__features = features
        self.__char = char
        self.__vector = vector

    def get_feature_val(self, feature: str):
        assert feature in self.__features, "Feature {} not in dimensions!"
        return self.__vector[self.__features.index(feature)]

    @property
    def vector(self):
        return self.__vector

    @property
    def char(self):
        return self.__char

    def __str__(self):
        s = self.__char + " ("
        for i, feature in enumerate(self.__features):
            if self.__vector[i] == 1:
                s += " " + self.__features[i]
        s += " )"
        return s


class Word(object):
    """ Wrapper class, contains a list of chars """

    def __init__(self, chars: List[Char]):
        self.__chars = chars

    def feature_array(self):
        """
        Agglutinates the feature vectors of the chars of the word
        Returns
            An numpy array of the size (n_chars x n_features)
        -------

        """
        return np.array([char.vector for char in self.__chars])

    @property
    def chars(self):
        return self.__chars

    def __str__(self):
        s = ""
        for char in self.__chars:
            s += char.char
        return s

    def __len__(self):
        return len(self.__chars)


class Alphabet(object):
    """ Class that manages the translation of cognate sets into vector arrays """
    # TODO: put re patterns in a file and load them
    match_long_vowel = re.compile(r"^[aɑɐeəɨiɪoɔœɒuʊɛyʏø]ː")
    match_nasal_vowel = re.compile(
        r"^[aɐeiwoɨuɑɔɛ]̃")  # not the best way to do it (nasal tilde not visible, use unicode instead?
    match_long_consonant = re.compile(r"^[pɸfbβvtθsdðzcɟçʝʃʒkxχgɣmnɲŋlɫʎrɾwɥ]ː")
    match_affricate = re.compile(r"^[tdɟ]͡[szʃʒʝ]")
    match_long_affricate = re.compile(r"^[tdɟ]͡[szʃʒʝ]ː")

    def __init__(self,
                 csv: Path,
                 encoding='utf-16',
                 pad_symbol="<pad>",
                 empty_symnol="-",
                 header_row=0,
                 chars_col=0,
                 ortho=False):
        self.__features = []
        self.__alphabet = []
        self.__dict = {}
        self.pad_symbol = pad_symbol
        self.empty_symbol = empty_symnol
        self.__header_row = header_row
        self.__chars_col = chars_col
        self.__encoding = encoding
        self.__ortho = ortho
        self._load(csv, encoding=self.__encoding)
        self.__char_embeddings = self.create_char_embeddings()

    def create_char_embeddings(self):
        """
        Creates character embeddings for all characters in the alphabet file
        Returns
        -------

        """
        char_embeddings = {}
        char2index = {char: i for i, char in enumerate(list(self.__dict.keys()))}

        for char, one_hot_vector in zip(char2index.keys(),
                                        keras.utils.to_categorical(list(char2index.values()),
                                                                   num_classes=len(char2index))):
            char_embeddings[char] = one_hot_vector
        return char_embeddings

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
        self._find_chars(word, chars_)
        chars = [self.create_char(char_) for char_ in chars_]
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
        self.__features = rows[self.__header_row].split(",")[1:]
        for row in rows[self.__header_row + 1:]:
            if row != "":
                cols = row.split(",")
                # check if the number of features for a given character matches
                # the total number of features
                if not self.__ortho:
                    assert len(cols) - 1 == len(self.__features), \
                        "Not enough features found, expected {}, got {}".format(len(self.__features), len(cols))
                char = cols[self.__chars_col]
                self.__alphabet.append(char)
                vec = []
                for feature_val in cols[self.__header_row + 1:]:
                    vec.append(int(feature_val))
                self.__dict[char] = vec
        # print(self._alphabet)

    def create_char(self, char: str):
        """
        Creates a Char object from a string
        Parameters
        ----------
        char
            The character from which the ASJP char should be translated
        Returns
            A Char object
        -------
        """
        assert char in self.__alphabet, "Unknown character '{}'".format(char)
        if not self.__ortho:
            return Char(char, self.__features, self.__dict[char])
        else:
            return Char(char, [], self.__char_embeddings[char])

    def get_char_by_vector(self, vector: np.array):
        """
        New method for both feature encodings and one-hot embeddings
        Parameters
        ----------
        vector
            The numpy array representing the features of the char
        Returns
            A Char object corresponding to the feature vector
        -------

        """
        cos_sims = {}
        if not self.__ortho:
            for c, v in self.__dict.items():
                cos_sims[c] = cos_sim(vector, v)
        else:
            for c, v in self.__char_embeddings.items():
                # only second dimension, the first one is lacking in the feature encodings
                # for unknown reasons
                cos_sims[c] = cos_sim(v, vector[0,])

        return max(cos_sims, key=cos_sims.get)

    def get_char_by_feature_vector(self, vec: np.array):
        """
        Finds the character whose feature vector/embedding is closest to the input vector.
        Atm we use cosine similarity to do the matching
        Parameters
        ----------
        vec
            The numpy array representing the features of the char
        Returns
            A char object corresponding to the feature vector
        -------
        """
        cos_sims = {}
        for c, feature_vector in self.__dict.items():
            cos_sims[c] = cos_sim(vec, feature_vector)

        return max(cos_sims, key=cos_sims.get)

    def get_char_by_embedding(self, vec: np.array):
        """
        Does the same as the method above, but for the one-hot-encoding
        Parameters
        ----------
        vec
            The numpy array representing the localist encoding of the car
        Returns
            A char object corresponding to the embedding
        -------
        """
        cos_sims = {}
        for c, feature_vector in self.__char_embeddings.items():
            cos_sims[c] = cos_sim(vec, feature_vector)

        return max(cos_sims, key=cos_sims.get)

    def get_feature_dim(self):
        """
        Determines the length of the feature set, i. e. the size of the input/output
        representations of the net. If encoding for features the dimensionality is the
        number of feaures.
        Returns
            The number of features
        -------

        """
        return len(self.__features)

    def get_embedding_dim(self):
        """
        Returns
            The size of the one-hot embeddings, which is the size of the alphabet, which
            is the size of the vocabulary.
        -------

        """
        return len(self)

    @property
    def char_embeddings(self):
        """
        Returns
            The dictionary mapping characters to their embeddings. Not used atm.
        -------

        """
        return self.__char_embeddings

    @property
    def feature_dim(self):
        if not self.__ortho:
            return len(self.__features)
        else:
            return len(self)

    def __str__(self):
        s = "*** ASJP alphabet class ***\n"
        if not self.__ortho:
            for ci, (char, vector) in enumerate(self.__dict.items()):
                feature_vals = "{}\t{}\t(".format(ci, char)
                for vi, val in enumerate(vector):
                    if int(val) == 1:
                        feature_vals += " {} ".format(self.__features[vi])
                feature_vals += ")"
                s += feature_vals + "\n"
        else:
            for ci, char in enumerate(self.__char_embeddings.keys()):
                s += char + "\n"
        return s

    def __len__(self):
        return len(self.__alphabet)


class CognateSet(object):
    """
    Models a cognate set in the data file. Provides the data structures required for
    the training phase, along with some convenience methods.
    """

    def __init__(self,
                 id: str,
                 concept: str,
                 ancestor: str,
                 cognate_dict: Dict[str, Word],
                 alphabet: Alphabet):
        self.__id = id
        self.__concept = concept
        self.__alphabet = alphabet
        self.__ancestor = ancestor
        self.__pad_to = max([len(word) for word in cognate_dict.values()])
        self.__cognate_dict = {lang: self._pad(word) for lang, word in cognate_dict.items()}

    @property
    def ancestor_word(self):
        """
        Returns
            The Word object representing the ancestor in the cognate set
        -------

        """
        return self.__cognate_dict[self.__ancestor]

    def _pad(self, word: Word):
        """
        Pads a word to an expected maximum length
        Parameters
        ----------
        word
            The word to pad
        Returns
            The padded word
        -------

        """
        for i in range(len(word), self.__pad_to):
            word.chars.append(self.__alphabet.create_char(self.__alphabet.empty_symbol))
        return word

    def __iter__(self):
        """ Mainly used for the training loops. yields an array of characters per location in the cognet set. """
        d = {}
        for i in range(self.__pad_to):
            for lang, word in self.__cognate_dict.items():
                d[lang] = word.chars[i].vector
            yield pd.DataFrame(data=d)

    def __str__(self):
        s = 5 * "=" + " Cognate Set {} ".format(self.id) + 5 * "=" + "\n"
        s += "concept: {}\n".format(self.concept)
        s += "ancestor: {}\n".format(self.__ancestor)
        for li, (lang, word) in enumerate(self.__cognate_dict.items()):
            if lang != self.__ancestor:
                s += "lang {}: {} {}\n".format(li, lang, word)
        s += "pad_to: {}\n".format(self.__pad_to)
        return s

    def __len__(self):
        return len(self.__cognate_dict)

    @property
    def id(self):
        return self.__id

    @property
    def concept(self):
        return self.__concept

    @property
    def ancestor(self):
        return self.__ancestor


class Ipa2Asjp(object):
    """ This class is used to convert the IPA data produced with epitran to asjp """

    def __init__(self,
                 sca,
                 ignored_symbols: List[str],
                 empty_symbol="-"):
        self.sca = sca
        self.empty_symbol = empty_symbol
        self.ignored_symbols = ignored_symbols

    def convert(self, chars: List[Char]):
        """
        Converts a list of (IPA) Char objects to their ASJP counterparts
        Parameters
        ----------
        chars
            The list of IPA Char objects
        Returns
            The list of ASJP Char objects
        -------

        """
        s = ""
        for char in chars:
            char_ = char.char
            if char_ == self.empty_symbol:
                s += self.empty_symbol
                continue
            # ignore start and stop symbols
            # if char_ == self.start_symbol or char_ == self.stop_symbol:
            #    continue
            # lingpy doesn't convert nasal vowels
            if char.get_feature_val('nasal') == 1 and char.get_feature_val('stop') == 0:
                char_ = char_.replace("ɐ̃", "ɐ"). \
                    replace("ɔ̃", "ɔ"). \
                    replace("w̃", "w"). \
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


class LevenshteinDistance(object):
    """ Uses the edit_distance function (levenshtein distance) to calculate our scores """

    def __init__(self,
                 true: List[str],
                 pred: List[str],
                 upper_bound=5):
        self.__true = true
        self.__pred = pred
        self.__word_lengths = [len(word) for word in self.__true]
        self.__upper_bound = upper_bound
        self.__distances = sorted([self._calculate_levenshtein(t, p) for t, p in zip(true, pred)], reverse=True)
        self.__mean_distance = np.mean(self.__distances)
        self.__mean_distance_normalized = self._mean_distance_norm()
        self.__percentiles = self._calculate_percentiles()

    def _calculate_levenshtein(self, t: str, p: str):
        """
        Calculates the Levenshtein distance between the latin word and the reconstructed word. Uses the
        Levenshtein distance function from NLTK.

        Parameters
        ----------
        t
            String representation of the latin word
        p
            String representation of the reconstructed word
        Returns
            The Levenshtein distance in characters
        -------
        """
        distance = nltk.edit_distance(t, p)
        return min(distance, self.__upper_bound)

    def _calculate_percentiles(self):
        """
        Calculates Levenshtein distance percentiles
        Returns
            A dictionary containing the percentiles
        -------

        """
        percentiles = {}
        data = Counter(self.__distances)

        # add up percentiles
        for distance, count in data.items():
            for percentile in percentiles:
                # if percentile > prev:
                percentiles[percentile] += count
            percentiles[distance] = count

        # divide by total number of distances
        percentiles = {percentile: count / len(self.__distances) for percentile, count in percentiles.items()}

        return percentiles

    def _mean_distance_norm(self):
        """
        Calculates the mean edit distance normalized by word length
        Returns
            The distance percentiles
        -------

        """
        normalized = []
        for length, distance in zip(self.__word_lengths, self.__distances):
            normalized.append(distance / length)
        return np.mean(normalized)

    def print_distances(self):
        print("Distances")
        data = Counter(self.__distances)
        for d, count in data.items():
            print("Distance={}: {}".format(d, count))
        print("Mean distance: {}".format(self.__mean_distance))
        print("Mean distance, normalized: {}".format(self.__mean_distance_normalized))

    def print_percentiles(self):
        print("Percentiles")
        for d, perc in self.__percentiles.items():
            print("Distance={}, {}".format(d, perc))

    @property
    def percentiles(self):
        return self.__percentiles

    @property
    def distances(self):
        return Counter(self.__distances)

    @property
    def mean_distance(self):
        return self.__mean_distance

    @property
    def mean_distance_normalized(self):
        return self.__mean_distance_normalized
