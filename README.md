# Automatic Die Studies for Ancient Numismatics

This repo contains the implementation of the paper *Automatic Die Studies for Ancient Numismatics*

[[`arXiv`](https://arxiv.org/abs/2407.20876)]

### Install

Clone with image matching submodules (including XFeat weights)
```
git clone --recurse-submodules <URL>
```

Then install dependencies
```
pip install -r requirements.txt
pip install -r accelerated_features/requirements.txt
cd RoMa
pip install .
```

### Add Datasets

To add a collection of coins, create a `datasets/<COLLECTION-NAME>` subfolder containing an image for each coin. Possible formats for the images are PNG or JPG. A subfolder should contain only coins that are likely to be struck by the same die. In particular reverse and obverse of the same coin should be in different subfolders. `<COLLECTION-NAME>` can be a symbolic link to a folder.

As some collections have images containing both obverse and reverse sides of coins, the Python utility function `utils.split_obverse_reverse` may be used to split in distinct directories.

<u>Facultative :</u> If clustering evaluation is needed (`clustering.evaluate_clustering`), add a `datasets/<COLLECTION-NAME>_Dies.xlsx` excel file, with columns `File` and `Die`. Utility `utils.create_labels_excel` may be used.

### Run an Automatic die study

Automatic die studies are ran from `run.py`, using the configuration described in `config.yaml`. Main options are:
- Matching Algorithm (XFeat | RoMa)
- Values of XFeat's $top_k$ and RoMa's confidence threshold
- Filtering (True | False)
- Dataset used (Paphos | Tanis1986 | Any added dataset)


### Example Scripts

Example scripts for computing similarities and clustering are to be found in `examples/`.
- `example_clustering.py` : Run and evaluate all 3 clustering methods on precomputed XFeat similarities, for the Paphos collection. 
- `example_similarities.py` : Compute similarities on a collection (both read from `config.yaml`). Save the results as `similarities/matches.npy`.


### Fully Automatic CADS Implementation

We provide two [Taylor's CADS](https://digitalcommons.trinity.edu/compsci_honors/54/) implementations (namely $\textit{CADS-AG}^*$ and $\textit{CADS-HDBSCAN}$) that are fully automatic. The corresponding Python functions are `cads_hdbscan` and `cads_ag_star` in the file `cads.py`.

### Reference
Clément Cornet, Héloïse Aumaître, Romaric Besançon, Julien olivier, Thomas Faucher, Hervé Le Borgne, [**Automatic Die Studies for Ancient Numismatics**](https://arxiv.org/abs/2407.20876), arXiv:2407.20876 , 2024 


```
@misc{cornet2024automaticdiestudiesancient,
      title={Automatic Die Studies for Ancient Numismatics}, 
      author={Clément Cornet and Héloïse Aumaître and Romaric Besançon and Julien Olivier and Thomas Faucher and Hervé Le Borgne},
      year={2024},
      eprint={2407.20876},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.20876}, 
}
```
