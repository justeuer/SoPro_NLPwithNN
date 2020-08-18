from pathlib import Path

from classes import Alphabet

id_col = 0
langs = ['latin', 'italian', 'spanish', 'french', 'portuguese', 'romanian']

path_to_ipa = Path("../data/alphabets/ipa.csv")
path_to_romance_data = Path("../data/romance_ipa_full.csv")

ipa = Alphabet(path_to_ipa)
romance_data = path_to_romance_data.open(encoding='utf-16').read().split("\n")
cols = romance_data[0].split(",")

romance_raw = [romance_data[i] for i in range(1, len(romance_data)-1) if i % 2 != 0]
romance_aligned = [romance_data[i] for i in range(2, len(romance_data)) if i % 2 == 0]

assert len(romance_raw) == len(romance_aligned), "aligned and raw data of different length: {}, {}"\
    .format(len(romance_raw), len(romance_aligned))

data = {}


for category, lines in {'raw': romance_raw, 'aligned': romance_aligned}.items():
    data[category] = {}
    for line in lines:
        if line == "":
            continue
        cognate_set = {}
        col_values = line.split(",")
        id = col_values[id_col]
        for col_name, col_value in zip(cols, col_values):
            if col_name in langs:
                word = ipa.translate(col_value)
                cognate_set[col_name] = word
        data[category][id] = cognate_set

errors = 0
for id, cognates in data['aligned'].items():
    l_lat = len(cognates['latin'].get_feature_array())
    for lang, word in cognates.items():
        ls = [len(word.get_feature_array()) for word in cognates.values()]
        for l in ls:
            if l != l_lat:
                errors += 1
                print(id, l_lat, ls, [char.get_char() for char in word.get_chars()])

if errors == 0:
    print("Everything is fine!")