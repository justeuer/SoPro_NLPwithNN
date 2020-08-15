from pathlib import Path
=======
import pandas as pd
import numpy as np
import tensorflow as tf
from ...classes.Alphabets import ASJPAlphabet
#from source.classes.Alphabets import ASJPAlphabet

cognates_file = Path("word_lists/cognates_romance.csv")

columns = ["ID", "CONCEPT", "LATIN", "ITALIAN", "SPANISH", "FRENCH", "ROMANIAN", "COMMENT"]

corpus = {}

corpus_set = []

def readFile(file):
	with cognates_file.open() as f:
		lines = f.read().split("\n")
		for li, line in enumerate(lines[1:len(lines)-1]):
			id = str(li+1)
			corpus[id] = {}
			row = line.split("\t")
			#print(row)
			for fi, field in enumerate(columns):
				column = columns[fi]
				corpus[id][column] = row[fi]

		data = pd.DataFrame.from_dict(corpus)
	return data

def transform(corpus):
	corpus = readFile(cognates_file)
	corpus = corpus['1']

	

transform(corpus_set)
