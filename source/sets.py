from pathlib import Path
import numpy as np
import re
from typing import Dict, List
import tensorflow as tf
import itertools
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


#we don't need this method anymore for right now--will keep just in case
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
        word_pairs = list(zip(lang_list,translation_list))
       # for l in zip(lang_list, translation_list):
            #word_pairs.append(list(l))
    #print("word pairs")
    #print(word_pairs)
    return zip(*word_pairs)


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

#method to convert the cognate set to an array
def translatedArray(path_to_asjp, cognate_set):
    asjp = Alphabet(path_to_asjp)
    aligned = asjp.translate_and_align(cognate_set)
    for lang, word in aligned.items():
        #print(lang, word)
        translated = word.get_feature_array()
    return translated



def load_dataset(cognate_set):
    """
    loads datasets and transform to tensors
    :param path:
    :param num_examples:
    :return:
    """
   # input_language, target_language = create_dataset(cognate_set)
   # print("target language")
   # print(target_language)

    '''
    For these methods, we only need the input tensor from the tokenizeTensor method, the input and
    target tokenizers from the tokenizeTokenizer method, and the target tensor from the translatedArray 
    method. The variables that aren't needed are referred to as dummy variables.
    '''
    # we only want the tensor from this method
    input_tensor, dummy_tokenizer = tokenizeTensor(cognate_set)
   # print("input tensor")
    #print(input_tensor)


    # we only want the tokenizer from this method
    dummy_tensor, input_tokenizer = tokenizeTokenizer(cognate_set)
    #print("input tokenizer")
    #print(input_tokenizer.word_index)

    #we only want the tokenizer from this method
    dummy_tensor, target_tokenizer = tokenizeTokenizer(cognate_set)
    #print("target tokenizer")
   # print(target_tokenizer.word_index)


    #we only want the tensor from this method
    target_tensor = translatedArray(path_to_asjp, cognate_set)
  #  print("target tensor")
   # print(target_tensor)
    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


#for debugging purposes
if __name__ == '__main__':
    load_dataset(cognate_set)
    #wordArray(path_to_asjp,cognate_set)
   #tokenizeNew(cognate_set)
  # tokenize(cognate_set)
