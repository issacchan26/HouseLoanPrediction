from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

input_file = "/Users/issacchan/PycharmProjects/data/loan_data.csv"
number_of_feature = 97
def df_preprocessing(df):
    for i in df:
        df[i] = df[i] / df[i].abs().max()
    return df

df = pd.read_csv(input_file, on_bad_lines='skip')
df = df[df.columns.drop(list(df.filter(regex='FLAG_DOCUMENT_')))]
df = df[df.columns.drop(list(df.filter(regex='EXT_SOURCE_')))]
df = df.drop(['SK_ID_CURR'], axis=1)
df = df.fillna(-1)

le = preprocessing.LabelEncoder()
columns = df.columns.values
for column in columns:
    if df[column].dtype != np.int64 and df[column].dtype != np.float64:
        df[column] = le.fit_transform(df[column].astype(str))

feature_name = df.drop(['TARGET'], axis=1).columns
df_target_1 = df.loc[df['TARGET'] == 1].reset_index(drop=True)
df_target_0 = df.loc[df['TARGET'] == 0].sample(n=df_target_1.shape[0])
df_sampled = pd.concat([df_target_1, df_target_0], ignore_index=True)
df_sampled = df_sampled.sample(frac=1).reset_index(drop=True)

X = df_sampled.drop(['TARGET'], axis=1).values
y = df_sampled['TARGET'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = XGBClassifier()
model.fit(X_train, y_train)
importance = model.feature_importances_

for j in range(1, number_of_feature):

    top = np.argpartition(importance, -j)[-j:]
    feature_name_selected = ['TARGET']
    for i in top:
        feature_name_selected.append(feature_name[i])
    processed_data = df_sampled[feature_name_selected]
    feature = processed_data.drop(['TARGET'], axis=1)
    feature = df_preprocessing(feature)
    X = feature.values
    y = processed_data['TARGET'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print(j, 'accuracy:', accuracy)


