from pathlib import Path
from typing import List


def list_to_str(lst: List[str]):
    s = ""
    for s_ in lst:
        s += "," + s_
    return s


cioabnu_ortho_loc = Path("../data/LREC-2014-parallel-list.txt")
cioabnu_ortho = cioabnu_ortho_loc.open(encoding='utf-8').read().split("\n")
ciobanu_corpus_file = Path("../data/romance_ciobanu_latin.csv")
ciobanu_corpus_file.touch()
ciobanu_corpus_file = ciobanu_corpus_file.open(mode='w', encoding='utf-16')


header = "id,concept,romanian,french,italian,spanish,portuguese,ancestor\n"
ciobanu_corpus_file.write(header)

for li, line in enumerate(cioabnu_ortho[2:]):
    if line == "":
        continue
    line = line.replace("<", "").replace(">", "")
    data = list_to_str(line.split())
    ciobanu_corpus_file.write("{},NA{}\n".format(li, data))

