from argparse import ArgumentParser
import epitran
from pathlib import Path


def remove_whitespaces(s: str):
    return " ".join(s.split())


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", type=str, default="../data/LREC-2014-parallel-list.txt")
    parser.add_argument("-o", type=str, default="../data/romance_ipa.csv")
    args = parser.parse_args()

    data_dir = Path(args.d)
    out_path = Path(args.o)

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
    for line in romance_data[1:len(romance_data)-1]:
        words = remove_whitespaces(line).split()
        s = ""
        for lang, word in zip(langs, words):
            if lang in transcriptors:
                ipa_word = transcriptors[lang].transliterate(word)
                "don't know why they have to use that letter"
                s += ipa_word + ","
        s = s[:len(s)-1] + "\n"
        #print(s)
        out_file.write(s)

    # test
    from classes import Alphabet
    ipa = Alphabet(Path("../data/alphabets/ipa.csv"))
    #print(ipa)
    test_str = "akt͡ʃelerat͡sie,akseleratjɔ̃,at͡ʃːelerasione,aseleɾasion".replace("ː", ":")
    test_words = test_str.split(",")
    for word in test_words:
        ipa_word = ipa.translate(word)
        print(ipa_word)
        print(ipa_word.get_feature_array())

if __name__ == "__main__":
    main()

