import math
import json
import logging

# General Imports
import pandas as pd

# SKLearn Imports

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay


# Load Data
df = pd.read_csv('../data/train.csv')
df.info()

df.head(5)

df.x6.unique()

df.x7.unique()

df_X = df.drop("y", axis=1)
df_label = df["y"]

df_X.head()


numeric_features = ["x1", "x2", "x4", "x5"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["x3", "x6", "x7"]
categorical_transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", LogisticRegression(max_iter=10000))]
)

clf


# Make LogReg Pipeline

RANDOM_STATE=1337

X_train, X_test, y_train, y_test = train_test_split(
    df_X,
    df_label,
    random_state=RANDOM_STATE
    )

print("model score: %.3f" % clf.score(X_test, y_test))


tprobs = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, clf.predict(X_test)))
print('Confusion matrix:')
print(confusion_matrix(y_test, clf.predict(X_test)))
print(f'AUC: {roc_auc_score(y_test, tprobs)}')
RocCurveDisplay.from_estimator(estimator=clf,X= X_test, y=y_test)