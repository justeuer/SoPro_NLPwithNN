from tensorflow import keras
from pathlib import Path

from utils import list_to_str

cioabnu_ortho_loc = Path("../data/LREC-2014-parallel-list.txt")
cioabnu_ortho = cioabnu_ortho_loc.open(encoding='utf-8').read()

latin_chars = set()

for line in cioabnu_ortho.split("\n")[2:]:
    latin_chars.update(set(line.replace(" ", "")))
    
# add missing characters
latin_chars.add("-")
latin_chars.add("œ")
# remove some spurious characters
latin_chars.remove("<")
latin_chars.remove(">")
latin_chars.remove(",")
print(latin_chars)

char_embeddings = {}
char2index = {char: i for i, char in enumerate(latin_chars)}
for char, one_hot_vector in zip(char2index.keys(),
                                keras.utils.to_categorical(list(char2index.values()),
                                num_classes=len(char2index))):
    char_embeddings[char] = one_hot_vector

assert len(char_embeddings) == len(latin_chars), "Size of embedding and character set mismatch, expected {}, got {}" \
    .format(len(latin_chars), len(char_embeddings))
print("Extracted char embeddings")


latin_path = Path("../data/alphabets/latin.csv")
latin_path.touch()
out = latin_path.open('w', encoding='utf-16')

# empty header
out.write("\n")

for char, embedding in char_embeddings.items():
    s = char + "," + list_to_str(embedding) + "\n"
    out.write(s)
out.close()

