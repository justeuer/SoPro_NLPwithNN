# Proto-word reconstruction with NNs

## How to train our models

### Python requirements
We used Python 3.7.9 in a conda environment. All required packages are listed in requirements.txt. To install them run:

    pip install -r requirements.txt
in your local environment.

We expect you to execute the scripts in the `source` folder.

### Command line parameters
This list can be queried in the command line by running a script with the `-h` flag.

| Parameter | Function | Default |
|--|--|--|
|`--aligned` | Use the rows containing the manual alignments | `False` |
|`--ancestor` | Header of the column containing the ancestor of the cognate set | `latin`
|`--data` | Path to the file containing the cognate sets | `../data/romance_swadesh_ipa.csv`|
|`--epochs` | Number of epochs | `10` |
|`--model`| One of [ipa, asjp, latin] | `ipa` |
|`--n_hidden` | Number of hidden layers in the feedforward model | `2` |
|`--ortho` | Use one-hot character embeddings  | `False` |
|`--out_tag` | Flag for the output folder | `swadesh` |

### Scripts
All Python scripts can be found in `scource/`. The scripts ending in `_cv.py` Perform cross-validation on the dataset
with 5 cross-validation folds. 
| Name | Description | Cross-validation |
|--|--|--|
|`feedforward.py` | Trains the feedforward model | No |
|`many2one_lstm.py` | Trains the LSTM model | No |
|`feedforward_cv.py` | Trains the feedforward model | Yes |
|`many2one_lstm_cv.py` | Trains the LSTM model | Yes |
**TODO: RNN**

### Sample script calls
* To train the feedforward model on ASJP feature encodings and the aligned data:

 `python feedforward.py --data=../data/romance_swadesh_ipa.csv --model=ipa --aligned --out_tag=swadesh`
 
 The results will be saved at `out/plots_swadesh_feedforward`.
* To train the LSTM model on Latin character embeddings on dataset **A**:

`python many2one_lstm.py --data../data/romance_ciobanu_latin.csv --model=latin --ortho`

The `--ortho` flag is required here since we don't have feature encodings for the Latin characters.

* To train the feedforward model on IPA character embeddings on dataset **B** and run cross-validation:

`python feedforward_cv.py --data=../data/romance_swadesh_ipa.csv --model=latin --ortho`

* **TODO: RNN**
