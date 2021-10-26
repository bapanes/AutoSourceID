# AutoSourceID

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4587205.svg)](https://doi.org/10.5281/zenodo.4587205)
[![DOI](https://doi.org/10.1051/0004-6361/202141193)
[![arXiv](https://img.shields.io/badge/arXiv-2103.11068-darkred.svg)](https://arxiv.org/abs/2103.11068)

Localization and classification of gamma-ray point sources using Machine Learning

![alt text](https://github.com/bapanes/Gamma-Ray-Point-Source-Detector/blob/main/figures/full-pipeline-high-lat-pie.png)

The material in this repoistory can be used to run a test example of the the pipeline developed in the paper [arXiv:2103.11068](https://arxiv.org/abs/2103.11068) in inference mode. Also, it includes routines to evaluate the test run and produce similar plots. Training algorithms for localization and classification can be used in combination with the data available at zenodo.org (see below).  

Codes to run patch generation, UNEK predictions and localization evaluations

```
from-cats-to-locnet-input.py

from-locnet-input-to-unek-output.py

from-unek-output-to-locnet-evaluation.py
```

Codes to visualize localization and classification results

```
localization-plots.ipynb

classification-plots.ipynb

full-pipeline-piechart.ipynb
```

Code to train classificatoin algorithm

```
classification-net-training.py
```

The data that is used to run the previous two codes can be found in the zenodo.org dataset associated to this repository, see details below

## ZENODO datasets for training and test

Along with this GitHub repository we also realease a dataset in the zenodo.org platform. This data includes thousand of patches, with their corresponding CSV files, which are useful to train new localization and classification algorithms. Also, we made available five blind data sets for tests, which can be used to make comparisons between different algorithms. All this material is available in the following link

[ZENODO page: Gamma-ray Point Sources Detection and Classification Project](https://zenodo.org/record/4587205#.YFOKBSPhD_Q)

Notice that some of the compressed files (.tgz) must be extracted by using the following command

```
tar -xjvf file_name.tgz 
```

