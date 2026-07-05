"""
model_training.py
------------------
Handles everything related to data preparation and machine learning
for the Iris Flower Classification project:

    1. Loading the dataset and converting it into a DataFrame.
    2. Splitting the data into training and testing sets.
    3. Scaling features with StandardScaler.
    4. Training multiple classification models.
    5. Comparing their accuracies and selecting the best one.
    6. Evaluating the best model in detail.
    7. Saving the trained model and scaler to disk.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from utils import (
    print_section_header,
    print_subsection,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_FILENAME,
    SCALER_FILENAME,
    TARGET_NAMES,
)


def load_iris_dataframe() -> pd.DataFrame:
    """
    Load the built-in scikit-learn Iris dataset and convert it into a
    single, human-readable Pandas DataFrame.

    The numeric target (0, 1, 2) is mapped to its species name
    (setosa, versicolor, virginica) in a 'species' column so that
    plots and printed tables are easy to interpret.

    Returns
    -------
    pd.DataFrame
        The Iris dataset with feature columns, a numeric 'target'
        column, and a readable 'species' column.
    """
    iris_bunch = load_iris()

    dataframe = pd.DataFrame(data=iris_bunch.data, columns=iris_bunch.feature_names)
    dataframe["target"] = iris_bunch.target
    dataframe["species"] = dataframe["target"].map(
        dict(enumerate(iris_bunch.target_names))
    )

    return dataframe


def explore_dataset(dataframe: pd.DataFrame) -> None:
    """
    Print a full exploratory summary of the dataset to the terminal:
    first rows, shape, feature/target names, missing values, and
    descriptive statistics.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The Iris dataset produced by `load_iris_dataframe`.

    Returns
    -------
    None
    """
    print_section_header("Dataset Overview")

    print_subsection("First 5 Rows")
    print(dataframe.head())

    print_subsection("Dataset Shape")
    print(f"Rows: {dataframe.shape[0]}, Columns: {dataframe.shape[1]}")

    print_subsection("Feature Names")
    print(list(dataframe.columns[:-2]))

    print_subsection("Target Names (Species)")
    print(TARGET_NAMES)

    print_subsection("Missing Values Per Column")
    print(dataframe.isnull().sum())

    print_subsection("Descriptive Statistics")
    print(dataframe.describe())


def split_and_scale_data(dataframe: pd.DataFrame):
    """
    Split the dataset into training and testing sets, then standardize
    the features using StandardScaler (fit only on the training data
    to avoid data leakage).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The Iris dataset produced by `load_iris_dataframe`.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print_section_header("Train/Test Split & Feature Scaling")

    feature_columns = [col for col in dataframe.columns if col not in ("target", "species")]
    X = dataframe[feature_columns].values
    y = dataframe["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Testing samples : {X_test_scaled.shape[0]}")
    print("Features standardized using StandardScaler (mean=0, std=1).")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_candidate_models() -> dict:
    """
    Build a dictionary of candidate machine learning models to train
    and compare.

    Returns
    -------
    dict
        Mapping of model name (str) to an unfitted scikit-learn
        estimator instance.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Train every candidate model on the training data, evaluate each
    on the test data, and return the fitted models along with their
    accuracy scores.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Scaled training and testing feature matrices.
    y_train, y_test : np.ndarray
        Training and testing target labels.

    Returns
    -------
    tuple
        (fitted_models: dict, accuracy_scores: dict)
    """
    print_section_header("Training & Comparing Multiple Models")

    candidate_models = get_candidate_models()
    fitted_models = {}
    accuracy_scores = {}

    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        fitted_models[model_name] = model
        accuracy_scores[model_name] = accuracy

        print(f"{model_name:<25} -> Accuracy: {accuracy * 100:.2f}%")

    return fitted_models, accuracy_scores


def select_best_model(fitted_models: dict, accuracy_scores: dict):
    """
    Identify the best-performing model based on test accuracy.

    Parameters
    ----------
    fitted_models : dict
        Mapping of model name to fitted estimator.
    accuracy_scores : dict
        Mapping of model name to test accuracy.

    Returns
    -------
    tuple
        (best_model_name: str, best_model: estimator)
    """
    best_model_name = max(accuracy_scores, key=accuracy_scores.get)
    best_model = fitted_models[best_model_name]

    print_section_header("Best Model Selected")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {accuracy_scores[best_model_name] * 100:.2f}%")

    return best_model_name, best_model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute detailed evaluation metrics for the given model on the
    test set and print a formatted classification report.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn classifier.
    X_test : np.ndarray
        Scaled test feature matrix.
    y_test : np.ndarray
        True test labels.
    model_name : str
        Name of the model, used for display purposes.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1_score,
        and the raw predictions (for confusion matrix plotting).
    """
    print_section_header(f"Detailed Evaluation — {model_name}")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print_subsection("Classification Report")
    print(classification_report(y_test, predictions, target_names=TARGET_NAMES))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": predictions,
    }


def save_model_and_scaler(model, scaler, model_path: str = MODEL_FILENAME, scaler_path: str = SCALER_FILENAME) -> None:
    """
    Persist the trained model and the fitted scaler to disk using
    Joblib, so that `prediction.py` can reload them later without
    retraining.

    Parameters
    ----------
    model : estimator
        The trained (best) scikit-learn model.
    scaler : StandardScaler
        The fitted feature scaler used during training.
    model_path : str
        Destination file path for the saved model.
    scaler_path : str
        Destination file path for the saved scaler.

    Returns
    -------
    None
    """
    print_section_header("Saving Model")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to  : {model_path}")
    print(f"Scaler saved to : {scaler_path}")