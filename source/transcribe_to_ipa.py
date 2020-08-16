from argparse import ArgumentParser
import epitran
import lingpy as lp
from pathlib import Path
from typing import List


def remove_whitespaces(s: str):
    return " ".join(s.split())


def lst_to_str(lst: List[str]):
    s = ""
    for c in lst:
        s += c
    return s


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", type=str, default="../data/LREC-2014-parallel-list.txt")
    parser.add_argument("-o", type=str, default="../data/romance_ipa.csv")
    parser.add_argument("--align", action='count', default=0)
    args = parser.parse_args()

    data_dir = Path(args.d)
    out_path = Path(args.o)
    align = bool(args.align)

    epi_french = epitran.Epitran('fra-Latn')
    epi_italian = epitran.Epitran('ita-Latn')
    epi_spanish = epitran.Epitran('spa-Latn')
    epi_romanian = epitran.Epitran('ron-Latn')
    epi_portuguese = epitran.Epitran('por-Latn')
    epi_latin = epitran.Epitran('lat-Latn')

    transcriptors = {
        'french': epi_french,
        'italian': epi_italian,
        'spanish': epi_spanish,
        'romanian': epi_romanian,
        'portuguese': epi_portuguese,
        'ancestor': epi_latin
    }

    out_path.touch()
    out_file = out_path.open('w')

    romance_data = data_dir.open().read().split("\n")

    langs = remove_whitespaces(romance_data[0]).split()
    langs = [lang.lower() for lang in langs]

    # build header
    header_str = ""
    for lang in langs:
        header_str += lang + ","
    header_str = header_str[:len(header_str)-1] + "\n"
    out_file.write(header_str)

    # Translate to IPA
    for line in romance_data[2:len(romance_data)-1]:
        words = remove_whitespaces(line).split()
        ipa_words = []
        for lang, word in zip(langs, words):
            if lang != 'ancestor':
                ipa_word = transcriptors[lang].transliterate(word)
                ipa_words.append(ipa_word)
            else:
                stripped_word = word.replace("io$", "one").\
                                    replace("o$", "ine").\
                                    replace("us$", "u").\
                                    replace("um$", "u").\
                                    replace("is$", "e").\
                                    replace("ns$", "nte").\
                                    replace("tas$", "tate").\
                                    replace("tor$", "tore").\
                                    replace("ter$", "tre")
                ipa_word = transcriptors['ancestor'].transliterate(stripped_word)
                ipa_words.append(ipa_word)

        # construct output string
        s = ""
        if align:
            aligned = lp.mult_align(ipa_words)
            for ipa_word in aligned:
                s += lst_to_str(ipa_word).replace("ː", ":") + ","
        else:
            for ipa_word in ipa_words:
                s += ipa_word.replace("ː", ":") + ","
        s = s[:len(s)-1] + "\n"
        out_file.write(s)

    # test
    from classes import Alphabet
    ipa = Alphabet(Path("../data/alphabets/ipa.csv"))
    test_str = "adad͡ʒiu,adaʒ,adad͡ʒ:o,adaxio,ɐdaʒio,adagɪʊm"
    test_words = test_str.split(",")
    ipa_words = []
    for word in test_words:
        ipa_word = ipa.translate(word)
        ipa_words.append(ipa_word)
        print(ipa_word)
        print(ipa_word.get_feature_array())


if __name__ == "__main__":
    main()

