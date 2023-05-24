# Neighborhood Relationship Networks

This repository contains the implementation of **Neighborhood Relationship Networks (NRN)**. It has been written in Python, using PyTorch.

Neighborhood Relationship Networks can learn the fine-grained dependencies for heterogeneous instances. Specifically, Neighborhood Encoder is designed to capture fine-grained semantics between neighbors, and Neighborhood Relationship Modules are stacked to achieve complex neighborhood relation modeling. 

## Installation

To install NRN, you can run:

```
git clone https://github.com/NRN-2023/Neighborhood-Relationship-Networks.git
```

### Environment Preparation

```
pip install -r requirements.txt
```

## Files Description

We implement the **NRN-M** and **NRN-A** networks proposed in the paper separately  at `NRN_MLP.py` and `NRN_Attention.py`.

Model training and imputation process code implemented in `imputation.py`.

## Running NRN

### Quickstart Example

You can run on the dataset `asf ` mentioned in our paper as an example：

```
python3 imputation.py
```

### Run on Your Dataset

You can add your dataset in the `./data/` folder. The schema of the data file is shown in `./data/asf.csv`.Then run:

```
python3 imputation.py --dataset 'your dataset name'
```

If you want to modify more parameters based on your dataset, you can refer to `imputation.py`.

