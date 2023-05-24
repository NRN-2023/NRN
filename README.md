# Neighborhood Relationship Networks

**Neighborhood Relationship Networks (NRN)** can learn the fine-grained dependencies for heterogeneous instances. Specifically, Neighborhood Encoder is designed to capture fine-grained semantics between neighbors, and Neighborhood Relationship Modules are stacked to achieve complex neighborhood relation modeling. 

We implement the **NRN-M** and **NRN-A** networks proposed in the paper separately  at `NRN_MLP.py` and `NRN_Attention.py`.Model training and imputation process code implemented in `imputation.py`.

## Installation

To install NRN, you can run:

```
git clone https://github.com/NRN-2023/Neighborhood-Relationship-Networks.git
```

### Environment Preparation

```
pip install -r requirements.txt
```

## Running NRN

### Quickstart Example

You can run on the dataset `asf ` mentioned in our paper as an example：

```
python3 imputation.py
```

### Run on your dataset

You can add your dataset in the `./data/` folder. The schema of the data file is shown in `./data/asf.csv`.Then run:

```
python3 imputation.py --dataset 'your dataset name'
```

If you want to modify more parameters based on your dataset, you can refer to `imputation.py`.

