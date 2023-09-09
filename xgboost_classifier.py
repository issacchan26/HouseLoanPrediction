from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

input_file = "path to train_data.csv"

def read_csv(path):
    df = pd.read_csv(path, on_bad_lines='skip')
    feature = df.drop(['TARGET'], axis=1)
    feature = preprocessing(feature)
    X = feature.values
    y = df['TARGET'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    return X_train, X_test, y_train, y_test

def preprocessing(df):
    for i in df:
        df[i] = df[i] / df[i].abs().max()
    return df

X_train, X_test, y_train, y_test = read_csv(input_file)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred, normalize=True)
print('accuracy:', accuracy)
