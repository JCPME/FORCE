# FORCE

This is the GitHub repository for the 2024 Deep Learning project at ETHZ.

## Project Structure

- **baselines/**
  - [logistic_regression.py](baselines/logistic_regression.py)
  - [MLP_feature_engineering.py](baselines/MLP_feature_engineering.py)
  - [README.md](baselines/README.md)

- **data/**
  - [readme.md](data/readme.md)

- **data_analysis/**
  - [dataset_analytics.py](data_analysis/dataset_analytics.py)
  - [readme.md](data_analysis/readme.md)

- **interpretability/**
  - [interpretability.py](interpretability/interpretability.py)
  - [README.md](interpretability/README.md)

- **model/**
  - [model.py](model/model.py)

- **Files:**
  - [requirements.txt](requirements.txt)
  - [requirements_pip.txt](requirements_pip.txt)
  - [.gitignore](.gitignore)
  - [README.md](README.md)

## Setup

 1: Clone the repository:
   ```sh
   git clone https://github.com/JCMCP/force.git
   cd force
  ```

2: Create a virtual environment
-  pip
```sh
python3 -m venv .venv
source .venv/bin/activate
```
```sh
pip install -r requirements_pip.txt
```
-  conda
```sh
conda env create -f environment.yml
conda activate dl
```

3: Download the dataset:\
   Download the dataset from (https://polybox.ethz.ch/index.php/s/VKQ8ELVzIfgBkP8) and put it in the data folder.\
   The password for the polybox is in the paper.

4: Run the scripts
```sh
python ./model/model.py
python ./baselines/logistic_regression.py
python ./baselines/MLP_feature_engineering.py
python ./data_analysis/dataset_analytics.py
python ./interpretability/interpretability.py

```


