# Predicting Gene Expression from Chromatin Landscape

_Project for course "Machine Learning for Genomics" at ETH._

## Setup

Dependencies for conda and pip are listed in `environment.yml` and `requirements.txt`.

The main notable requirement is `python=3.9.0` due to `3.10` not working with some other requirements.
`environment.yml` contains my whole setup, however, most is macOS Metal specific libraries.

Every python dependency is listed in `requirement.txt` without nested dependencies.
`tensorflow-macos` and `tensorflow-metal` are also system specific dependencies, and varies for other systems
([installation guide](https://www.tensorflow.org/install/pip)).
 
Most reliable way to replicate environment:
```commandline
conda create -n gene_exp_rnn python=3.9
conda activate gene_exp_rnn
pip install -r requirements.txt
```
(..and install tensorflow according to guide above and remove other tensorflow dependencies if not using Metal.)

### Data
The [project data](https://polybox.ethz.ch/index.php/s/iY6d8qbMMiy4dQh) should be unzipped into `./data`. For example, the X1 dataset train info should be available at `./data/CAGE-train/X1_train_info.tsv`.

### Notable dependencies
- Python 3.9.0
- Histone modification data processing with [pyBigWig](https://github.com/deeptools/pyBigWig).
