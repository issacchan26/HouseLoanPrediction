# House Loan Prediction with XGBoost and MLP
Data Analysis and prediction on Kaggle dataset: House Loan Data Analysis-Deep Learning  
This repo provides algorithms for data preprocessing and model training for House Loan problem  
Dataset: https://www.kaggle.com/datasets/deependraverma13/house-loan-data-analysis-deep-learning

## Data Preprocessing
There are 122 columns in original dataset, including groundtruth ('TARGET' column), this repo used 20 feature columns and 1 'TARGET' column for groundtruth to train the model  
Before running [read_csv.py](read_csv.py), please change the path of 'loan_data (1).csv' in line 7 to your local path  
It will output two files to your current directory: train_data.csv and balanced_data.csv  
The data preprocessing will present in both [read_csv.py](read_csv.py) and [train.py](train.py)

## Model Training
Please change the path of train_data.csv in line 10 to your local path  
The hyperparameters used in [train.py](train.py):  
epoch = 10000  
batch_size = 4  
learning_rate=0.0001  
optimizer: SGD

## Inference
Please change the path of train_data.csv and state_dict model to your local path  
[test.py](test.py) will output the accuracy and mean loss after forward pass the validation dataset to your model

## XGBoost
[xgboost_classifier.py](xgboost_classifier.py) provides the comparison between XGBoost and MLP method, using same set of training data  
Please change the path of train_data.csv to your local path  
[end_to_end_xgboost.py](end_to_end_xgboost.py) provides the end to end script for XGBoost, by taking loan_data.csv as input  
It will output all accuracy results for each number of features





