##################################################################################################
### Should test the behaviour of our code
##################################################################################################

from Constants import asjp_char_dims, asjp_char_as_vector
from classes.Characters import ASJPChar
from classes.Alphabets import ASJPAlphabet
from classes.Words import Word
import tensorflow as tf
import numpy as np

def check_vector_dims():

    for i, (char, vec) in enumerate(asjp_char_as_vector.items()):
        assert len(vec) == len(asjp_char_dims), \
            "Feature vector in column {} ({}) not of appropriate length!" \
                .format(i, char)

    print("Vector dims match!")

def check_char_vectorizers():
    for char, vec in asjp_char_as_vector.items():
        asjp_char = ASJPChar(char)
       # print(asjp_char)

def test_ASJP_alphabet():
    word = "blau"
    asjp = ASJPAlphabet()
    word_list = ['o:s', 'os[so]', '(we)s[o]', 'os', 'os']
    for word in word_list:
        translated = asjp.translate(word)
        array = translated.get_feature_array()
        array_list = []
        array_list.append(array)
   # for array in array_list:
        #print("word array")
      # print(array)
    translated = asjp.translate(word)
   # print(translated)
    array = translated.get_feature_array()
    array = tf.convert_to_tensor(array)
  #  print(type(array))
    '''Added by Morgan'''
    #this will save a single tensor
    #save the array to a file
   # np.save("test", array, allow_pickle=False)
    #open the file to make sure everything worked
   # print(np.load("test.npy"))


def main():
   # print("Testing....")
   # check_vector_dims()
   # check_char_vectorizers()
    test_ASJP_alphabet()

if __name__ == "__main__":
    main()
