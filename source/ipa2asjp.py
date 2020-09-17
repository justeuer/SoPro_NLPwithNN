from lingpy import rc
from pathlib import Path

from classes import Ipa2Asjp, Alphabet

ipa = Alphabet(Path("../data/alphabets/ipa.csv"))
sca = rc('asjp')
converter = Ipa2Asjp(sca, ["ː"])

romance_ipa_path = Path("../data/romance_ipa_auto.csv")
romance_ipa = romance_ipa_path.open(encoding='utf-16').read()

out_path = Path("../data/romance_asjp_auto.csv")
out_path.touch()
out_file = out_path.open('w', encoding='utf-16')

langs = ["latin", "italian", "spanish", "french", "portuguese", "romanian"]
col_names = ["id", "concept"] + langs

header = "id,concept,latin,italian,spanish,french,portuguese,romanian\n"
out_file.write(header)
print(header)

for line in romance_ipa.split("\n")[1:]:
    s = ""
    if line != "":
        row = line.split(",")
        assert len(row) == len(col_names), "Expected {} fields, found {}"\
            .format(len(col_names), row)
        # create row data dict
        row_data = {col_name: row[col_names.index(col_name)] for col_name in col_names}
        s += row_data['id']
        s += "," + row_data['concept']
        for lang in langs:
            w = row_data[lang]
            # We are not sure were some of these chars slip in (some are in the original data), but for our pipeline
            # they have to be removed.
            w = w.replace(":", "").replace("ː", "").replace("ʁ", "").replace("ɡ", "g").replace("ā", "a")\
            .replace('ă', "").replace('ῑ', "i").replace("é", "").replace('ș', "").replace('ŭ', "").replace("í", "").replace("ý", "")\
            .replace('ĭ', "").replace('š', "").replace("á", "a").replace("è", "").replace("â", "")\
            .replace("̃", "").replace('̆', "").replace("́", "").replace('̄', "").replace("ʷ", "").replace('̈', "").replace('̂', "")\
            .replace("ț", "").replace('͡', "").replace("ɬ", "").replace('̌', "").replace("<", "").replace(">", "").replace("2", "")
            ipa_w = ipa.translate(w)
            asjp_w = converter.convert(ipa_w.chars)
            # construct string
            s += "," + asjp_w
        s += "\n"
        print(s)
        out_file.write(s)

out_file.close()
