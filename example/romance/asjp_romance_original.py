from pathlib import Path

cognates_file = Path("word_lists/cognates_romance.csv")

columns = ["ID", "CONCEPT", "LATIN", "ITALIAN", "SPANISH", "FRENCH", "ROMANIAN", "COMMENT"]

corpus = {}

with cognates_file.open() as f:
    lines = f.read().split("\n")
    for li, line in enumerate(lines[1:len(lines)-1]):
        id = str(li+1)
        corpus[id] = {}
        row = line.split("\t")
        print(row)
        for fi, field in enumerate(columns):
            column = columns[fi]
            corpus[id][column] = row[fi]

print("ID\tCONCEPT\tLATIN\tITALIAN\tSPANISH\tFRENCH\tROMANIAN\n")
for id, cognate_set in corpus.items():
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(id, cognate_set["ID"], cognate_set["CONCEPT"], cognate_set["LATIN"],
                                                cognate_set["SPANISH"], cognate_set["FRENCH"], cognate_set["ROMANIAN"],
                                                cognate_set["COMMENT"]))
