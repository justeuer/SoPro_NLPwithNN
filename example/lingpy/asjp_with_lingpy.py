##################################################################################################
### ASJP:
###     https://asjp.clld.org/
###     Word lists: https://asjp.clld.org/languages
###     Introduced by Brown (2008), see papers
### ASJP Sound class model:
###     https://en.wikipedia.org/wiki/Automated_Similarity_Judgment_Program
### LingPy: Python library for historical linguists:
###     http://lingpy.org/
###     Module reference: https://lingpy.readthedocs.io/en/latest/reference/lingpy.html
### Pyclts:
###     https://pypi.org/project/pyclts/
###     neat tool if we want to generate transcriptions (fast) and don't want to use lingpy
###     for that. However, lingpy is preferred.
##################################################################################################

import copy
from pathlib import Path
from lingpy import rc

asjp = rc('asjp')
print(asjp)

# columns
DESCRIPTION = 4
LOAN = 6
WORD = 8
CONCEPT1 = 9
CONCEPT2 = 10

working_dir = Path("./word_lists")

word_lists = {}

for lang in ['fr', 'sp', 'it', 'lat']:
    word_lists[lang] = {}
    file = working_dir / (lang + ".csv")
    with open(file.absolute(), 'r', encoding='utf-8') as f:
        raw = f.read().split("\n")
        for _, line in enumerate(raw[1:len(raw)-1]):
                columns = line.split(",")
                desc = columns[DESCRIPTION].split("-")
                if int(desc[2]) == 1:
                    concept = desc[1]
                    loan = False if columns[LOAN] == "False" else True
                    word = columns[WORD]
                    word_lists[lang][concept] = tuple((concept, loan, word))

cognates = {}
for lang in ['fr', 'sp', 'it', 'lat']:
    for _, data in word_lists[lang].items():
        concept = data[0]
        loan = data[1]
        word = data[2]
        if not loan:
            if concept in cognates.keys():
                if lang not in cognates[concept].keys():
                    cognates[concept][lang] = word
            else:
                cognates[concept] = {lang: word}

cognates_purged = copy.deepcopy(cognates)
for concept, cognate_set in cognates.items():
    if len(cognate_set.items()) < 4:
        cognates_purged.pop(concept)

print("Full cognate sets found: {}".format(len(cognates_purged)))

print(asjp.converter['b'])

