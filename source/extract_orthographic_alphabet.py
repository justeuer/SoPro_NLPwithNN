from tensorflow import keras
from pathlib import Path

from utils import list_to_str

ID_COL = "id"
CONCEPT_COL = "concept"

# load data

romance_ortho_loc = Path("../data/romance_orthographic.csv")
print("Load romance orthographic data from {}".format(romance_ortho_loc.absolute()))
romance_ortho = romance_ortho_loc.open(encoding='utf-16').read().split("\n")
print("Loaded {} cognate sets".format(len(romance_ortho[1:])))

cols = romance_ortho[0].split(",")
langs = cols[2:]

latin_chars = set()

for line in romance_ortho[1:]:
    if line == "":
        break
    col_data = line.split(",")
    for col_name, data in zip(cols, col_data):
        if col_name in langs:
            for char in data:
                latin_chars.add(char)

# add special chars
latin_chars.add('<start>')
latin_chars.add('<stop>')
latin_chars.add('<pad>')
latin_chars.add('-')

print("Extracted {} chars : {}".format(len(latin_chars), latin_chars))

# create char embeddings

char_embeddings = {}
char2index = {char: i for i, char in enumerate(latin_chars)}
for char, one_hot_vector in zip(char2index.keys(),
                                keras.utils.to_categorical(list(char2index.values()),
                                num_classes=len(char2index))):
    char_embeddings[char] = one_hot_vector

assert len(char_embeddings) == len(latin_chars), "Size of embedding and character set mismatch, expected {}, got {}"\
    .format(len(latin_chars), len(char_embeddings))
print("Extracted char embeddings")

# save to file

latin_path = Path("../data/alphabets/latin.csv")
latin_path.touch()
out = latin_path.open('w', encoding='utf-16')

# empty header
out.write("\n")

for char, embedding in char_embeddings.items():
    s = char + "," + list_to_str(embedding) + "\n"
    out.write(s)
out.close()


