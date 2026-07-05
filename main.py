"""
main.py
-------
Entry point for the Iris Flower Classification project.

Running this script executes the full machine learning pipeline:

    1. Load and explore the Iris dataset.
    2. Perform exploratory data analysis with professional plots.
    3. Split and scale the data.
    4. Train and compare five different classification models.
    5. Automatically select the best-performing model.
    6. Evaluate that model in detail and plot its confusion matrix.
    7. Plot feature importance (if supported by the best model).
    8. Save the trained model and scaler to disk for later use in
       prediction.py.

Usage
-----
    python main.py
"""

from model_training import (
    load_iris_dataframe,
    explore_dataset,
    split_and_scale_data,
    train_and_compare_models,
    select_best_model,
    evaluate_model,
    save_model_and_scaler,
)
from visualization import (
    plot_pairplot,
    plot_correlation_heatmap,
    plot_boxplots,
    plot_histograms,
    plot_class_distribution,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
)
from utils import print_section_header, TARGET_NAMES, FEATURE_NAMES


def run_pipeline() -> None:
    """
    Execute the complete Iris Flower Classification pipeline from
    start to finish.

    Returns
    -------
    None
    """
    print_section_header("Iris Flower Classification Project")
    print("An end-to-end Machine Learning pipeline built with Scikit-learn.")

    # 1. Load dataset and convert to DataFrame.
    dataframe = load_iris_dataframe()

    # 2. Display dataset overview (head, shape, features, targets,
    #    missing values, descriptive statistics).
    explore_dataset(dataframe)

    # 3. Exploratory Data Analysis — generate all professional plots.
    print_section_header("Exploratory Data Analysis")
    plot_pairplot(dataframe)
    plot_correlation_heatmap(dataframe)
    plot_boxplots(dataframe)
    plot_histograms(dataframe)
    plot_class_distribution(dataframe)
    print("All EDA plots saved successfully as PNG files.")

    # 4. Split into train/test sets and scale features.
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(dataframe)

    # 5. Train multiple models and compare their accuracies.
    fitted_models, accuracy_scores = train_and_compare_models(
        X_train, X_test, y_train, y_test
    )
    plot_accuracy_comparison(accuracy_scores)

    # 6. Automatically select the best-performing model.
    best_model_name, best_model = select_best_model(fitted_models, accuracy_scores)

    # 7. Evaluate the best model in detail.
    evaluation_results = evaluate_model(best_model, X_test, y_test, best_model_name)

    # 8. Plot the confusion matrix for the best model.
    plot_confusion_matrix(
        y_test,
        evaluation_results["predictions"],
        class_names=TARGET_NAMES,
        model_name=best_model_name,
    )

    # 9. Plot feature importance for the best model. Tree-based models
    # expose feature_importances_ natively, linear models expose coef_.
    # For models with neither (e.g. an RBF-kernel SVM or KNN), fall back
    # to permutation importance so feature_importance.png is always produced.
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(
            FEATURE_NAMES, best_model.feature_importances_, best_model_name
        )
    elif hasattr(best_model, "coef_"):
        import numpy as np
        mean_abs_coefficients = np.mean(np.abs(best_model.coef_), axis=0)
        plot_feature_importance(FEATURE_NAMES, mean_abs_coefficients, best_model_name)
    else:
        from sklearn.inspection import permutation_importance
        print("\nSelected model has no built-in importance scores -- computing permutation importance instead.")
        permutation_result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42)
        plot_feature_importance(FEATURE_NAMES, permutation_result.importances_mean, best_model_name)

    # 10. Save the trained model and scaler for use in prediction.py.
    save_model_and_scaler(best_model, scaler)

    print_section_header("Pipeline Completed Successfully")
    print(f"Best Model : {best_model_name}")
    print(f"Accuracy   : {evaluation_results['accuracy'] * 100:.2f}%")
    print("Run 'python prediction.py' to classify a new flower sample.\n")


if __name__ == "__main__":
    run_pipeline()