"""
prediction.py
-------------
Interactive command-line tool that lets a user enter Iris flower
measurements and get a predicted species using the model trained and
saved by main.py.

Usage
-----
    python prediction.py

Make sure you have already run `python main.py` at least once so that
'saved_model.pkl' and 'saved_scaler.pkl' exist in the project folder.
"""

import os
import sys

import joblib
import numpy as np

from utils import (
    print_section_header,
    print_subsection,
    TARGET_NAMES,
    FEATURE_NAMES,
    MODEL_FILENAME,
    SCALER_FILENAME,
)


def load_trained_model_and_scaler(model_path: str = MODEL_FILENAME, scaler_path: str = SCALER_FILENAME):
    """
    Load the previously trained model and scaler from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    scaler_path : str
        Path to the saved scaler file.

    Returns
    -------
    tuple
        (model, scaler)

    Raises
    ------
    SystemExit
        If the model or scaler file does not exist, with a helpful
        message instructing the user to run main.py first.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print_section_header("Model Not Found")
        print(
            "Could not find a saved model. Please run 'python main.py' first "
            "to train and save the model before making predictions."
        )
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def get_float_input(prompt: str) -> float:
    """
    Prompt the user for a single numeric measurement and keep asking
    until a valid, positive float is entered.

    Parameters
    ----------
    prompt : str
        The message displayed to the user.

    Returns
    -------
    float
        A valid, positive floating-point measurement in centimeters.
    """
    while True:
        raw_value = input(prompt).strip()
        try:
            value = float(raw_value)
            if value <= 0:
                print("  -> Please enter a positive number greater than 0.")
                continue
            if value > 50:
                print("  -> That value seems unusually large for a flower measurement (cm). Please re-check.")
                continue
            return value
        except ValueError:
            print("  -> Invalid input. Please enter a numeric value (e.g., 5.1).")


def collect_measurements_from_user() -> np.ndarray:
    """
    Collect all four flower measurements from the user via the
    terminal, handling invalid input gracefully.

    Returns
    -------
    np.ndarray
        A 2D array of shape (1, 4) ready to be scaled and fed into
        the model, in the order defined by FEATURE_NAMES.
    """
    print_subsection("Enter Flower Measurements (in centimeters)")

    sepal_length = get_float_input("Sepal Length (cm): ")
    sepal_width = get_float_input("Sepal Width  (cm): ")
    petal_length = get_float_input("Petal Length (cm): ")
    petal_width = get_float_input("Petal Width  (cm): ")

    measurements = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return measurements


def predict_species(model, scaler, measurements: np.ndarray) -> None:
    """
    Scale the given measurements, run them through the trained model,
    and print the predicted species along with class probabilities
    (when supported by the model).

    Parameters
    ----------
    model : estimator
        The trained scikit-learn classifier.
    scaler : StandardScaler
        The fitted scaler used during training.
    measurements : np.ndarray
        A 2D array of shape (1, 4) containing the raw measurements.

    Returns
    -------
    None
    """
    scaled_measurements = scaler.transform(measurements)
    predicted_label = model.predict(scaled_measurements)[0]
    predicted_species = TARGET_NAMES[predicted_label]

    print_section_header("Prediction Result")
    print(f"Predicted Species: {predicted_species.upper()}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_measurements)[0]
        print_subsection("Class Probabilities")
        for species_name, probability in zip(TARGET_NAMES, probabilities):
            print(f"  {species_name:<12}: {probability * 100:.2f}%")


def run_prediction_loop() -> None:
    """
    Main interactive loop: repeatedly ask the user for measurements
    and print predictions until they choose to stop.

    Returns
    -------
    None
    """
    print_section_header("Iris Flower Species Predictor")
    print("Enter flower measurements to predict its species.")
    print(f"Expected features: {', '.join(FEATURE_NAMES)}")

    model, scaler = load_trained_model_and_scaler()

    while True:
        measurements = collect_measurements_from_user()
        predict_species(model, scaler, measurements)

        again = input("\nPredict another flower? (y/n): ").strip().lower()
        if again != "y":
            print("\nThank you for using the Iris Flower Species Predictor. Goodbye!\n")
            break


if __name__ == "__main__":
    try:
        run_prediction_loop()
    except KeyboardInterrupt:
        print("\n\nPrediction interrupted by user. Exiting gracefully.")
        sys.exit(0)