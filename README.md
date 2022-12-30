# MLOps - final project

## structure 
    
    .
    ├── datasets_utils.py           # Loads the datasets from the relevant CSV.
    ├── pipeline_utils.py           # builds and runs the pipeline.
    ├── base_models.py              # Holds the base model as used per each dataset.
    ├── README.md
    ├── datasets
    │    ├── freMPTL2freq.csv       # French Motor claims dataset.
    │    ├── housing.csv            # (optional) Boston housing prices dataset.
    │
    └─ 

## Install
(from the project directory)
```shell
conda env create -f environment.yml
```

if that doesn't work, use commends as follows:
```shell
conda create --name mlops-project python=3.8.15 matplotlib=3.1.1
conda activate mlops-project
conda install -c conda-forge xgboost=1.5.0
pip install macest
```