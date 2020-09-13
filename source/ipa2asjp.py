from lingpy import rc
from pathlib import Path

from classes import Asjp2Ipa, Alphabet

ipa = Alphabet(Path("../data/alphabets/ipa.csv"))
sca = rc('asjp')
converter = Asjp2Ipa(sca, ["ː"])

romance_ipa_path = Path("../data/romance_ipa_full.csv")
romance_ipa = romance_ipa_path.open(encoding='utf-16').read()

out_path = Path("../data/romance_asjp_full.csv")
out_path.touch()
out_file = out_path.open('w', encoding='utf-16')

langs = ["latin", "italian", "spanish", "french", "portuguese", "romanian"]
col_names = ["id", "concept"] + langs

header = "id,concept,latin,italian,spanish,french,portuguese,romanian\n"
out_file.write(header)
print(header)

for line in romance_ipa.split("\n")[1:201]:
    s = ""
    row = line.split(",")
    assert len(row) == len(col_names), "Expected {} fields, found {}"\
        .format(len(col_names), row)
    # create row data dict
    row_data = {col_name: row[col_names.index(col_name)] for col_name in col_names}
    s += row_data['id']
    s += "," + row_data['concept']
    for lang in langs:
        w = row_data[lang]
        ipa_w = ipa.translate(w)
        asjp_w = converter.convert(ipa_w.get_chars())
        # construct string
        s += "," + asjp_w
    s += "\n"
    print(s)
    out_file.write(s)

out_file.close()