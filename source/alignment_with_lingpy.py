import lingpy as lp
from pathlib import Path
from typing import List

from classes import Alphabet


def lst_to_str(lst: List[str]):
    s = ""
    for c in lst:
        s += c
    return s


ipa_csv_path = Path("../data/alphabets/ipa.csv")
ipa = Alphabet(ipa_csv_path)
romance_data_path = Path("../data/romance_ipa_partial.csv")
romance_data = romance_data_path.open(encoding='utf-16').read()
out_path = Path("../data/romance_ipa_aligned.csv")
out_file = out_path.open('w')

langs = ["latin", "italian", "spanish", "french", "romanian"]

header = "id,concept,latin,italian,spanish, french,romanian\n"

out_file.write(header)

cognate_sets = {}
for line in romance_data.split("\n")[1:51]:
    s = ""
    data = line.split(",")
    id = str(data[0])
    concept = data[1]
    s += id + "," + concept
    cognate_sets[id] = {}
    # align
    aligned = lp.mult_align(data[2:])
    for lang, w in zip(langs, aligned):
        cognate_sets[id][lang] = w
        s += "," + lst_to_str(w)
    s += "\n"
    out_file.write(s)

