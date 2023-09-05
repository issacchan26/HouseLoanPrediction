from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

input_file = "path to loan_data.csv"
pd.set_option('display.max_columns', None)

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

top20 = np.argpartition(importance, -20)[-20:]
feature_name_selected = ['TARGET']
print('Top 20 features:')
for i in top20:
    feature_name_selected.append(feature_name[i])

print(top20)
processed_data = df_sampled[feature_name_selected]
print(processed_data)

# output = {'features': feature_name, 'importance': importance}
# output = pd.DataFrame(output)
processed_data.to_csv('./train_data.csv', index=False, header=True)
df_sampled.to_csv('./balanced_data.csv', index=False, header=True)