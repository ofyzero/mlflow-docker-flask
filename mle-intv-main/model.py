import logging
import matplotlib.pyplot as plt
from pandas import DataFrame
from tabulate import tabulate

# General Imports
import pandas as pd
import numpy as np
import yaml 
import os 
import shutil
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

# SKLearn Imports

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay

logger = logging.getLogger(__name__)

CONFIG_FILE = "./config.yaml"

mlflow.set_experiment("mle-intv")


@dataclass
class Config:
    input_data: str
    numeric_features: list
    categorical_features: list
    random_state: int
    model_output: str


def load_config(path):
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            logger.info('Configuration loaded successfully.')
            return config
        except yaml.YAMLError as exc:
            logger.error('Failed to load configuration.', exc_info=True)
            raise exc

# Load Data
def load_data(path):
   
    df = pd.read_csv(path)

    print(df.info()) 
    print(df.head(5))
    print(df.x6.unique())
    print(df.x7.unique())

    return df


def pipeline(numeric_features, categorical_features): 

    
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

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

    print(clf)

    return clf


def evaluation(clf, X_test, y_test):

    # Log model score
    model_score = clf.score(X_test, y_test)
    print("Model score: %.3f" % model_score)
    
    mlflow.log_metric("model_score", model_score)

    # Predict probabilities
    tprobs = clf.predict_proba(X_test)[:, 1]

    # Log classification report
    clf_report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    print("Classification report:")
    print(classification_report(y_test, clf.predict(X_test)))

    # Assuming clf_report is a dictionary
    df = DataFrame(clf_report).transpose()

    # Convert DataFrame to tabular format
    _ = tabulate(df, headers='keys', tablefmt='grid',  showindex=True)

    # Plot DataFrame
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # Hide axis
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()

    # Save plot as PNG image
    plt.savefig('./models/clf_report.png')

    mlflow.log_artifact("./models/clf_report.png")

    # Log confusion matrix
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    print("Confusion matrix:")
    print(conf_matrix)

    # Save the confusion matrix to a file
    np.savetxt('./models/confusion_matrix.txt', conf_matrix)
    mlflow.log_artifact("./models/confusion_matrix.txt")

    # Log AUC score
    auc_score = roc_auc_score(y_test, tprobs)
    print("AUC:", auc_score)

    mlflow.log_metric("AUC", auc_score)
    
    # Log ROC curve
    roc_display = RocCurveDisplay.from_estimator(estimator=clf, X=X_test, y=y_test)

    # Save ROC curve figure to a file
    roc_display.figure_.savefig("./models/ROC_curve.png")

    # Log the ROC curve figure as an artifact with MLflow
    mlflow.log_artifact("./models/ROC_curve.png")


def save_model(clf, model_path):
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    mlflow.sklearn.save_model(clf, model_path)


def main(args_data):
    
    #load data
    df = load_data(args_data.input_data)

    #drop target
    df_X = df.drop("y", axis=1)
    
    #target
    df_label = df["y"]
    
    print(df_X.head())

    #pipeline variables
    numeric_features = args_data.numeric_features
    categorical_features = args_data.categorical_features

    # Make Pipeline
    clf = pipeline(numeric_features, categorical_features)

    # Split Data
    RANDOM_STATE=args_data.random_state
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_label, random_state=RANDOM_STATE)

    print(X_test.shape, type(X_test))
    print(X_test.head())

    # Make LogReg Pipeline
    with mlflow.start_run():
        
        # Train your model
        clf.fit(X_train, y_train)

        # Log parameters, metrics, and model
        mlflow.log_param("numeric_features", numeric_features)
        mlflow.log_param("categorical_features", categorical_features)
        mlflow.log_param("RANDOM_STATE", RANDOM_STATE)
        # mlflow.log_metric("metric_name", metric_value)
        mlflow.sklearn.log_model(clf, "model_name")
    
        # Evaluate
        evaluation(clf, X_test, y_test)

        # Save Model
        save_model(clf, args_data.model_output)


if __name__ == "__main__":

    # Load Config
    config = load_config(CONFIG_FILE)

    args_data = Config(**config["general"])

    main(args_data)