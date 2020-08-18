from pathlib import Path
import numpy as np
import re
from typing import Dict, List
import tensorflow as tf

'''
This script preprocesses the cognate sets from asjp (curated from the wordlists found on https://asjp.clld.org/),
tokenizes them, etc, to prepare them to be loaded into the neural network 
'''

path_to_asjp = Path("/home/morgan/Documents/saarland/fourth_semester/nn_software_project/sopro-nlpwithnn/data/alphabets/asjp.csv")

word_feature_list = []
lang_list = []
translation_list = []
word_pairs = []

cognate_set = {
        "lat": "o:s",
        "it": "os[so]",
        "sp": "(we)s[o]",
        "fr": "os",
        "rom": "os"
    }

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


class Alphabet(object):
    """ Class that manages the translation of cognate sets into vector arrays """
    match_parentheses = re.compile(r'^\(.*\)')
    match_square_brackets = re.compile(r'^\[.*\]')
    match_vowel = re.compile(r'^[aeiouE3]:')

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
            elif bool(self.match_vowel.match(chunk)):
                group = self.match_vowel.match(chunk).group(0)
                chunk_ = chunk[len(group):]
                chars.append(group)
                self._find_chars(chunk_, chars)
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
        aligned_languages = {}

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
                if bool(self.match_vowel.match(chunk)) and len(chunk) == 2:
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
        assert char in self._alphabet, "Unknown character {}".format(char)
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


#method to create the corpus (a nested list of each element of the cognate set dictionary with 'start'
#and 'stop' markers
def create_dataset(cognate_set):
    #need to get items from cognate set dictionary to create the corpus
    for language, translation in cognate_set.items():
        #create start and stop markers for both sides of the corpus
        '<start> ' + language + ' <end>'
        language_marker = '<start> ' + language + ' <end>'
        translation_marker = '<start> ' + translation + ' <end>'
        lang_list.append(language_marker)
        translation_list.append(translation_marker)
        for l in zip(lang_list, translation_list):
            word_pairs.append(list(l))
    #print("word pairs")
  #  print(word_pairs)
    return zip(*word_pairs)

#method to tokenize the created corpus
def tokenize(language):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(language)
    tensor = tokenizer.texts_to_sequences(language)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer

#method to convert the cognate set to an array
def wordArray(path_to_asjp, cognate_set):
    asjp = Alphabet(path_to_asjp)
   # input_language, target_language = create_dataset(cognate_set)
   # print(target_language)
    aligned = asjp.translate_and_align(cognate_set)
   # print(aligned)
    for input_language, target_language in aligned.items():
        input_language, input_tensor = tokenize(input_language)
       # print("input language")
       # print(input_language)
       # print("input tensor")
      #  print(input_tensor)
        translated = target_language.get_feature_array()
        target_tensor = tf.convert_to_tensor(translated)
        #print("input tensor shape")
       # print(input_tensor.shape)
       # print("target tensor shape")
       # print(target_tensor.shape)
       # word_feature_list.append(word.get_feature_array())
        #TODO: this might need padding later (see code from n2c2_expression.py) but not sure since
        #with the translation schema used, all arrays are the same length
        #print(word_feature_list)
        return input_language, target_language, input_tensor, target_tensor


def load_dataset(cognate_set):
    """
    loads datasets and transform to tensors
    :param path:
    :param num_examples:
    :return:
    """
    input_language, target_language = create_dataset(cognate_set)
    input_tensor, input_tokenizer = tokenize(input_language)
    target_tensor, target_tokenizer = tokenize(target_language)
    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


#for debugging purposes
#if __name__ == '__main__':
   # load_dataset(cognate_set)
   # create_dataset(cognate_set)
  #  wordArray(path_to_asjp,cognate_set)
