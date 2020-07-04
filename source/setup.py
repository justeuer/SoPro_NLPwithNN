##################################################################################################
### Should test the behaviour of our code
##################################################################################################

from classes.Vectorizers import asjp_cons_as_vector
from classes.Characters import asjp_cons_dims
from classes.Characters import ASJPChar

def check_vector_dims():

    for i, (char, vec) in enumerate(asjp_cons_as_vector.items()):
        assert len(vec) == len(asjp_cons_dims), \
            "Feature vector in column {} ({}) not of appropriate length!" \
                .format(i, char)

    print("Vector dims match!")

def check_char_vectorizers():
    for char, vec in asjp_cons_as_vector.items():
        asjp_char = ASJPChar(char)
        print(asjp_char, vec)

def main():
    print("Testing....")
    check_vector_dims()
    check_char_vectorizers()

if __name__ == "__main__":
    main()
