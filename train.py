import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from model import MLP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

path = 'path to train_data.csv'
epoch = 10000
batch_size = 8
learning_rate=0.00001

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

X_train, X_test, y_train, y_test = read_csv(path)
train_tensor = TensorDataset(X_train, y_train) 
train_loader = DataLoader(dataset = train_tensor, batch_size = batch_size)
test_tensor = TensorDataset(X_test, y_test) 
test_loader = DataLoader(dataset = test_tensor, batch_size = batch_size)
accuracy_loader = DataLoader(dataset = test_tensor, batch_size = 1)  # batch size = 1 for accuracy calculation

device = torch.device('cuda')
writer = SummaryWriter()  # tensorboard for monitoring the training progress
model = MLP(50,1).to(device)
criterion = nn.BCELoss()  # BCE for binary classification
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if os.path.exists('./best.pt'):  # Check if there are previous best.pt model, the model will resume training if model exists
    model.load_state_dict(torch.load('./best.pt'))

best_loss = 99999

for i in range(0, epoch+1):
    train_loss = 0
    val_loss = 0
    acc = 0

    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        output = model(X_batch)
        loss = criterion(output, y_batch.unsqueeze(1))         
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_pred = model(X_val)
            loss = criterion(y_pred, y_val.unsqueeze(1)) 
            val_loss += loss.item()           

        if i % 100 == 0:
            for X_val, y_val in accuracy_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                if y_pred.round() == y_val:
                    acc += 1

            print('----------------------------------------------')
            print('epoch:', i)
            print('train_loss:', train_loss)
            print('val_loss:', val_loss) 
            print('mean_accuracy:', acc/len(accuracy_loader))
            writer.add_scalar('mean accuracy', acc/len(accuracy_loader), i)
    
    writer.add_scalar('train loss', train_loss, i)
    writer.add_scalar('val loss', val_loss, i)

    torch.save(model.state_dict(), 'latest.pt')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best.pt')
    
    
