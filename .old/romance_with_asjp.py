from copy import deepcopy
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed, cpu_count

# columns
DESCRIPTION = 4
LOAN = 6
WORD = 8
CONCEPT1 = 9
CONCEPT2 = 10

n_cores = cpu_count() - int(cpu_count() / 4)
print("{} cores available".format(cpu_count()))
print("Using {} cores!".format(n_cores))

working_dir = Path("../example/lingpy/word_lists")
save_dir = Path("../data")
if not save_dir.exists():
    save_dir.mkdir()

word_lists = {}

for lang in ['fr', 'sp', 'it', 'lat']:
    word_lists[lang] = {}
    file = working_dir / (lang + ".csv")
    with open(file.absolute(), 'r', encoding='utf-8') as f:
        raw = f.read().split("\n")
        for _, line in enumerate(raw[1:len(raw) - 1]):
            columns = line.split(",")
            desc = columns[DESCRIPTION].split("-")
            if int(desc[2]) == 1:
                concept = desc[1]
                loan = False if columns[LOAN] == "False" else True
                word = columns[WORD]
                word_lists[lang][concept] = tuple((concept, loan, word))

cognates = {}
for lang in ['fr', 'sp', 'it', 'lat']:
    for _, romance in word_lists[lang].items():
        concept = romance[0]
        loan = romance[1]
        word = romance[2]
        if not loan:
            if concept in cognates.keys():
                if lang not in cognates[concept].keys():
                    cognates[concept][lang] = word
            else:
                cognates[concept] = {lang: word}

cognates_purged = deepcopy(cognates)
for concept, cognate_set in cognates.items():
    if len(cognate_set.items()) < 4:
        cognates_purged.pop(concept)

print("Full cognate sets found: {}".format(len(cognates_purged)))

latin = {
    concept: cognate_set["lat"]
    for concept, cognate_set in cognates_purged.items()
}

romance = {
    concept: {
        lang: word for lang, word in cognate_set.items() if lang != 'lat'
    }
    for concept, cognate_set in cognates_purged.items()
}

print(latin)
print(romance)

from classes.Alphabets import ASJPAlphabet

asjp = ASJPAlphabet()

latin = {concept: asjp.translate(word) for concept, word in latin.items()}


def save_as_numpy(out: Path, arr: np.array):
    np.save(out.absolute(), arr, allow_pickle=False)


lat_dir = save_dir / "lat"
if not lat_dir.exists():
    lat_dir.mkdir()

with Parallel(n_jobs=n_cores) as parallel:
    parallel(delayed(save_as_numpy)(lat_dir / ("lat_" + concept), word.get_feature_array())
             for concept, word in latin.items())

for file in lat_dir.iterdir():
    array = np.load(file.absolute())
    print(file.stem, '\n', array)
