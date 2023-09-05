# Kaggle dataset: House Loan Data Analysis-Deep Learning
Data Analysis and prediction on Kaggle dataset: House Loan Data Analysis-Deep Learning  
This repo provides algorithms for data preprocessing and model training for House Loan problem  
Dataset: https://www.kaggle.com/datasets/deependraverma13/house-loan-data-analysis-deep-learning

## Data Preprocessing
There are 122 columns in original dataset, including groundtruth ('TARGET' column), this repo used 20 feature columns and 1 'TARGET' column for groundtruth to train the model  
Before running [preprocess.py](preprocess.py), please change the path of 'loan_data (1).csv' in line 7 to your local path  
It will output two files to your current directory: train_data.csv and balanced_data.csv

## Model Training
Please change the path of train_data.csv in line 10 to your local path  
The hyperparameters used in [train.py](train.py):  
epoch: 10000  
batch_size: 16  
learning_rate: 0.0000001  
optimizer: SGD

## Inference
Please change the path of train_data.csv and state_dict model to your local path  
[test.py](test.py) will output the accuracy and mean loss after forward pass the validation dataset to your model




