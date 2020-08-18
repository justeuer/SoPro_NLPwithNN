from pathlib import Path
import numpy as np
import re
from typing import Dict, List


class Char(object):
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
    def __init__(self, chars: List[Char]):
        self.chars = chars

    def get_feature_array(self):
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
    #match_parentheses = re.compile(r'^\(.*\)')
    #match_square_brackets = re.compile(r'^\[.*\]')
    match_long_vowel = re.compile(r"^[aɑɐeəɨiɪoɔœɒuʊɛyʏø]ː")
    match_nasal_vowel = re.compile(r"^[aɐeiwoɨuɑɔɛ]̃") # not the best way to do it (nasal tilde not visible, use unicode instead?
    match_long_consonant = re.compile(r"^[pɸfbβvtθsdðzcɟçʝʃʒkxχgɣmnɲŋlɫʎrɾwɥ]ː")
    match_affricate = re.compile(r"^[tdɟ]͡[szʃʒʝ]")
    match_long_affricate = re.compile(r"^[tdɟ]͡[szʃʒʝ]ː")

    def __init__(self,
                 csv: Path,
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
        self._load(csv)

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
        to_align = {}
        for lang, word in cognates.items():
            chars_ = []
            self._find_chars(word, chars_)
            to_align[lang] = chars_

        return self._align_cognates(to_align)

    def translate(self, word: str):
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
            """
            if bool(self.match_parentheses.match(chunk)):
                group = self.match_parentheses.match(chunk).group(0)
                chunk_ = chunk[len(group):]
                group = group[1:len(group) - 1]
                chars.append(group)
                self._find_chars(chunk_, chars)
            elif bool(self.match_square_brackets.match(chunk)):
                group = self.match_square_brackets.match(chunk).group(0)
                chunk_ = chunk[len(group):]
                group = group[1:len(group) - 1]
                # ignore affixation
                for _ in range(len(group)):
                    chars.append(self.empty_symbol)
                self._find_chars(chunk_, chars)
            """

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

    def _align_cognates(self, cognates: Dict[str, List[str]]):
        """
        Aligns a set of cognates split up in chunks
        Parameters
        ----------
        cognates
            A dict with language ids as keys and lists of chunks as values
        Returns
            A dict containing the aligned cognate words per language
        -------
        """
        aligned_chunks = {lang: [] for lang in cognates}
        aligned_words = {}

        # collect chunks at each postion, replace empty chunks with the empty symbol
        max_l_word = max([len(chars) for chars in cognates.values()])
        for i in range(max_l_word):
            chunks_at_i = {}
            for lang, chunks in cognates.items():
                if i < len(chunks):
                    chunks_at_i[lang] = chunks[i]
                else:
                    chunks_at_i[lang] = self.empty_symbol

            # split aligned chunks into the alphabet symbols
            max_l_chunk = max([len(chunk_at_i) for chunk_at_i in chunks_at_i.values()])
            for lang, chunk in chunks_at_i.items():
                # take care of long vowels
                if bool(self.match_long_vowel.match(chunk)) and len(chunk) == 2:
                    chunk_ = chunk
                    aligned_chunks[lang].append(chunk_)
                    aligned_chunks[lang].append(self.empty_symbol)
                elif len(chunk) < max_l_chunk:
                    chunk_ = chunk
                    aligned_chunks[lang].append(chunk_)
                    for _ in range(max_l_chunk - len(chunk)):
                        aligned_chunks[lang].append(self.empty_symbol)
                elif len(chunk) > 1:
                    for chunk_ in chunk:
                        aligned_chunks[lang].append(chunk_)
                else:
                    chunk_ = chunk
                    aligned_chunks[lang].append(chunk_)

        # create Word object with aligned strings
        for lang, aligned_list in aligned_chunks.items():
            chars_ = [self._create_char(self.start_symbol)]
            for aligned_chunk in aligned_list:
                chars_.append(self._create_char(aligned_chunk))
            chars_.append(self._create_char(self.stop_symbol))
            aligned_words[lang] = Word(chars_)

        return aligned_words

    def _load(self, path: Path):
        """
        Loads the feature set for a classes.Alphabets.Alphabet from a csv file
        Parameters
        ----------
        path
            The path to the csv file
        -------
        """
        rows = path.open(encoding='utf-16').read().split("\n")
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


if __name__ == '__main__':
    cognate_set = {
        "lat": "o:s",
        "it": "os[so]",
        "sp": "(we)s[o]",
        "fr": "os",
        "rom": "os"
    }

    path_to_asjp = Path("../data/alphabets/asjp.csv")
    asjp = Alphabet(path_to_asjp)
    aligned = asjp.translate_and_align(cognate_set)

    for lang, word in aligned.items():
        print(lang, word)
        print(word.get_feature_array())

    path_to_ipa = Path("../data/alphabets/ipa.csv")
    ipa = Alphabet(path_to_ipa)
    print(ipa)
    ipa_word = ipa.translate("fra:tɛr")
    for char in ipa_word.chars:
        print(char.get_char(), char.get_feature_vector())
