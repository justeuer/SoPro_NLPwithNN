from pathlib import Path
import numpy as np
import re
from typing import Dict, List
import tensorflow as tf
import itertools
import pandas as pd
from tensorflow import keras
'''
This script preprocesses the cognate sets from asjp (curated from the wordlists found on https://asjp.clld.org/),
tokenizes them, etc, to prepare them to be loaded into the neural network 
'''

path_to_asjp = Path("/home/morgan/Documents/saarland/fourth_semester/nn_software_project/sopro-nlpwithnn/data/alphabets/asjp.csv")

path_to_ipa = Path("/home/morgan/Documents/saarland/fourth_semester/nn_software_project/sopro-nlpwithnn/data/alphabets/asjp.csv")

path_to_ipa_sets = Path("/home/morgan/Documents/saarland/fourth_semester/nn_software_project/sopro-nlpwithnn/data/alphabets/romance_ipa_full.csv")

path_to_asjp_sets = Path("/home/morgan/Documents/saarland/fourth_semester/nn_software_project/sopro-nlpwithnn/data/romance_asjp_full.csv")

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


class Word(object):
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
    match_nasal_vowel = re.compile(
        r"^[aɐeiwoɨuɑɔɛ]̃")  # not the best way to do it (nasal tilde not visible, use unicode instead?
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
        self._char_embeddings = self.create_char_embeddings()

    def create_char_embeddings(self):
        """
        Creates a
        Returns
        -------

        """
        char_embeddings = {}
        char2index = {char: i for i, char in enumerate(list(self._dict.keys()))}

        for char, one_hot_vector in zip(char2index.keys(),
                                        keras.utils.to_categorical(list(char2index.values()),
                                                                   num_classes=len(char2index))):
            char_embeddings[char] = one_hot_vector
        return char_embeddings

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

    def get_char_embeddings(self):
        return self._char_embeddings

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
        self._features = rows[self.header_row].split(",")[1:]
        for row in rows[self.header_row + 1:]:
            if row != "":
                cols = row.split(",")
                assert len(cols) - 1 == len(self._features), \
                    "Not enough features found, expected {}, got {}".format(len(self._features), len(cols))
                char = cols[self.chars_col]
                self._alphabet.append(char)
                vec = []
                for feature_val in cols[self.header_row + 1:]:
                    vec.append(int(feature_val))
                self._dict[char] = vec
       # print(self._alphabet)

    def create_char(self, char: str):
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


class CognateSet(object):
    def __init__(self,
                 id: str,
                 concept: str,
                 ancestor: str,
                 cognate_dict: Dict[str, Word],
                 alphabet: Alphabet,
                 pad_to: int):
        self.id = id
        self.concept = concept
        self.alphabet = alphabet
        self.ancestor = ancestor
        self.pad_to = pad_to
        self.cognate_dict = {lang: self._pad(word) for lang, word in cognate_dict.items()}

    def _pad(self, word: Word):
        for i in range(len(word), self.pad_to):
            word.get_chars().append(self.alphabet.create_char(self.alphabet.pad_symbol))
        return word

    def __iter__(self):
        d = {}
        for i in range(self.pad_to):
            for lang, word in self.cognate_dict.items():
                d[lang] = word.get_chars()[i].get_feature_vector()
            yield pd.DataFrame(data=d)

    def __str__(self):
        s = 5*"=" + " Cognate Set {} ".format(self.id) + 5*"=" + "\n"
        s += "concept: {}\n".format(self.concept)
        s += "ancestor: {}\n".format(self.ancestor)
        for li, lang in enumerate(self.cognate_dict.keys()):
            if lang != self.ancestor:
                s += "lang {}: {}\n".format(li, lang)
        s += "pad_to: {}\n".format(self.pad_to)
        return s


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




'''
So for our net to work, we need to have an equally shaped input language and target language, as well as a vocabulary size.
The current method is to use the values from the cognate set dictionary (the cognate words,
transcribed into ASJP) as the input and the resulting array from translating the input into ASJP as the target input.
For the input language, the result of creating a tokenizer object would be an object of shape of around (3,1) (after converting this to an array) (I don't know the exact shape).
For the target language, the result of creating an array would be an array of shape of (7,27).
So, for the input language, we have to manually pad the array (after converting it from a tokenizer object to an array) so that it matches the shape of the target language,
since the input language and target language arrays are being computed by different methods. Therefore, we cannot simply
use the padding library provided by Keras because it can't pad for both the input language and the target language. So,
we had to pad manually, which can be found in the tokenizeTensor method.

Additionally, the net requires a vocabulary size, which is computed from the tokenizer object with the "word_index" function. 
For the input language, this was easily computed as our raw data was in string format, so we simply just had to use the method provided
by TensorFlow in order to encode the data and return a tokenizer object. 
For the translate language, this was not as easily computed because the raw input was an array of shape (7,27).
So, for right now, we have both the input vocabulary and the target vocabulary as coming from the same source (the transcribed cognate words
from the cognate set dictionary). For this we use the tokenizeTokenizer method, which returns a tokenizer object.
For the input tensor, we use the tokenizeTensor method that returns an array, padded to match the shape of the translated array (7.27).
Finally, for the target tensor, we use the translatedArray method to compute the translations from the transcribed values into an array.

'''

def extractASJP(path_to_asjp):
    cols = ['id', 'concept', 'latin', 'italian', 'spanish', 'french', 'portuguese', 'romanian']
    langs = cols[2:]
    asjp = Alphabet(Path(path_to_asjp), encoding='utf-8')
    asjp_sets = Path(path_to_asjp_sets)
    asjp_data = asjp_sets.open(encoding='utf-16').read().split("\n")
    # purge unaligned data
    romance_aligned = [line for i, line in enumerate(asjp_data[1:]) if i % 2 == 1]
    romance_aligned = romance_aligned[0]
    line_split = romance_aligned.split(",")
    assert len(line_split) == len(cols), "Not enough values in line: Expected {}, got {}".format(len(cols),
                                                                          len(line_split))
    #create cognate dict for sets
    cognate_dict = {}
    for word, col in zip(romance_aligned.split(","), cols):
        if col in langs:
            cognate_dict[col] = asjp.translate(word)

    cs = CognateSet(id="1",
                    concept="I",
                    ancestor='latin',
                    cognate_dict=cognate_dict,
                    alphabet=asjp,
                    pad_to=5)

    for datapoint in cs:
        #target is the tensor for latin
        target = datapoint.pop(cs.ancestor)
        #datapoint is the tensor for the input of the five
        #other romance languages


    #add padding to the target tensor (latin) to make it match the
    #input of the five romance languages
    padded_target = np.zeros(datapoint.shape)
    padded_target[:datapoint.shape[0],:datapoint.shape[1]] = datapoint
    return datapoint, padded_target

#method to get a tokenizer object for both the input and target tokenizers to be able to compute a vocabulary
#list for both input and target languages
def tokenizeTokenizer(cognate_set):
    translate = list(cognate_set.values())[0]
    #we need to have markers to mark the start of the transcription and the end of the transcription
    #since we are doing this on the character level, this part isn't exactly necessary but it is necessary
    #to get it working with the neural network that we are using
    translate_with_markers = '< ' + translate + ' >'
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(translate_with_markers)
    tensor_tokenizer = tokenizer.texts_to_sequences(translate_with_markers)
    return tensor_tokenizer, tokenizer


#method to get the input tensor, padded to match the shape of the target tensor
def tokenizeTensor(cognate_set):
    #get the first value in the cognate set dictionary (need to figure out how to get multiple values)
    translate = list(cognate_set.values())[0]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(translate)
    tensor_tokenizer = tokenizer.texts_to_sequences(translate)
    #tensor_tokenizer = tf.keras.preprocessing.sequence.pad_sequences(tensor_tokenizer, padding='post')
    #print(tensor_tokenizer)
    # set the padding parameters to later add padding (we want a size of (7,27)
    max_len = 27
    for i in tensor_tokenizer:
        if len(i) > max_len:
            max_len = len(i)
    # create the padding
    tensor = np.array([i + [0] * (max_len - len(i)) for i in tensor_tokenizer])
    # convert the padded tensor to an array so we can change the
    # dimensions and add more padding
    tensor_array = np.array(tensor)
    # add enough axes to the array (7) to be equivalent to the
    # ASJP translation
    empty_array = np.zeros((4,27))
    tokenized_array = np.vstack((tensor_array, empty_array))
    return tokenized_array, tensor_tokenizer


#def load_dataset(path_to_asjp):
 #   """
 #   loads datasets and transform to tensors
 #   :param path:
 #   :param num_examples:
 #   :return:
 #   """


  #  return input_tensor, input_tokenizer, target_tensor, target_tokenizer


#for debugging purposes
if __name__ == '__main__':
  #  load_dataset(cognate_set)
    #wordArray(path_to_asjp,cognate_set)
   #tokenizeNew(cognate_set)
  # tokenize(cognate_set)
  extractASJP(path_to_asjp)
