import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from model import MLP
from torch.utils.data import TensorDataset, DataLoader

path = 'path to train_data.csv'
model_path = 'path to your state_dict model'

def read_csv(path):
    df = pd.read_csv(path, on_bad_lines='skip')
    feature = df.drop(['TARGET'], axis=1)
    feature = preprocessing(feature)
    X = feature.values  # extract the features from input df
    y = df['TARGET'].values  # extract the ground truth from input df
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)  # eval dataset
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)  # eval dataset

    return X_train, X_test, y_train, y_test

def preprocessing(df):  # preprocessing the data to the range of (-1,1)
    for i in df:
        df[i] = df[i] /df[i].abs().max()
    return df

_, X_test, _, y_test = read_csv(path)
test_tensor = TensorDataset(X_test, y_test) 
test_loader = DataLoader(dataset = test_tensor, batch_size = 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(20,1).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCELoss()

acc = 0
val_loss = 0
total_0 = 0
total_1 = 1
tar_0 = 0
tar_1 = 0

model.eval()
with torch.no_grad():
    for X_val, y_val in test_loader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        y_pred = model(X_val)
        loss = criterion(y_pred, y_val.unsqueeze(1)) 
        val_loss += loss

        if y_val == 0:
            total_0 += 1
            if y_pred.round() == y_val:
                tar_0 += 1
                acc += 1

        elif y_val == 1:
            total_1 += 1
            if y_pred.round() == y_val:
                tar_1 += 1
                acc += 1

acc = acc/len(test_loader)
val_loss = val_loss/len(test_loader)

print('total number of class 0:', total_0)
print('correct number of class 0:', tar_0)
print('total number of class 1:', total_1)
print('correct number of class 1:', tar_1)
print('accuracy:', acc)
print('mean loss:', val_loss.item())

