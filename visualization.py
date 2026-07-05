"""
visualization.py
----------------
All plotting functions for the Iris Flower Classification project.

Every function saves a high-resolution (300 DPI) PNG file to disk so
that the graphs are available for the README "Screenshots" section
and for reviewers to inspect without re-running the code.
"""

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend so the script
                       # never blocks waiting for a plot window to close.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import print_subsection

# A consistent, professional color palette used across every chart.
PALETTE = {
    "setosa": "#4C72B0",
    "versicolor": "#DD8452",
    "virginica": "#55A868",
}
SNS_PALETTE = list(PALETTE.values())

# Apply a clean, professional global style once for the whole project.
sns.set_theme(style="whitegrid", context="talk")


def plot_pairplot(dataframe: pd.DataFrame, output_path: str = "pairplot.png") -> None:
    """
    Create and save a Seaborn pairplot showing pairwise relationships
    between all four Iris features, colored by species.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The Iris dataset including a 'species' column.
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating pairplot.png")

    plot_columns = [col for col in dataframe.columns if col != "target"]
    grid = sns.pairplot(
        dataframe[plot_columns],
        hue="species",
        palette=PALETTE,
        diag_kind="hist",
        height=2.2,
        plot_kws={"alpha": 0.75, "edgecolor": "white", "linewidth": 0.4},
    )
    grid.fig.suptitle(
        "Pairwise Feature Relationships by Species", y=1.02, fontsize=18, fontweight="bold"
    )
    grid.fig.set_size_inches(14, 12)
    grid.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def plot_correlation_heatmap(
    dataframe: pd.DataFrame, output_path: str = "correlation_heatmap.png"
) -> None:
    """
    Create and save a correlation heatmap of the numeric Iris features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The Iris dataset (numeric feature columns only are used).
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating correlation_heatmap.png")

    numeric_data = dataframe.select_dtypes(include=[np.number]).drop(columns=["target"], errors="ignore")
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    )
    plt.title("Feature Correlation Heatmap", fontsize=18, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplots(dataframe: pd.DataFrame, output_path: str = "boxplots.png") -> None:
    """
    Create and save boxplots for each feature grouped by species, to
    visualize the spread and outliers of each measurement.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The Iris dataset including a 'species' column.
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating boxplots.png")

    feature_columns = [col for col in dataframe.columns if col not in ("species", "target")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for index, feature in enumerate(feature_columns):
        sns.boxplot(
            data=dataframe,
            x="species",
            y=feature,
            hue="species",
            palette=PALETTE,
            legend=False,
            ax=axes[index],
        )
        axes[index].set_title(f"{feature.title()} by Species", fontsize=14, fontweight="bold")
        axes[index].set_xlabel("Species", fontsize=12)
        axes[index].set_ylabel(feature.title(), fontsize=12)

    fig.suptitle("Feature Distributions by Species (Boxplots)", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(dataframe: pd.DataFrame, output_path: str = "histograms.png") -> None:
    """
    Create and save histograms for each numeric feature to visualize
    their overall distribution shape.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The Iris dataset including a 'species' column.
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating histograms.png")

    feature_columns = [col for col in dataframe.columns if col not in ("species", "target")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for index, feature in enumerate(feature_columns):
        sns.histplot(
            data=dataframe,
            x=feature,
            hue="species",
            palette=PALETTE,
            kde=True,
            element="step",
            ax=axes[index],
        )
        axes[index].set_title(f"Distribution of {feature.title()}", fontsize=14, fontweight="bold")
        axes[index].set_xlabel(feature.title(), fontsize=12)
        axes[index].set_ylabel("Frequency", fontsize=12)

    fig.suptitle("Feature Histograms by Species", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    dataframe: pd.DataFrame, output_path: str = "class_distribution.png"
) -> None:
    """
    Create and save a bar chart showing how many samples belong to
    each Iris species (class balance check).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The Iris dataset including a 'species' column.
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating class_distribution.png")

    counts = dataframe["species"].value_counts().reindex(PALETTE.keys())

    plt.figure(figsize=(10, 7))
    bars = plt.bar(counts.index, counts.values, color=SNS_PALETTE, edgecolor="black")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            str(int(height)),
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    plt.title("Class Distribution of Iris Species", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Species", fontsize=13)
    plt.ylabel("Number of Samples", fontsize=13)
    plt.ylim(0, max(counts.values) + 10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_comparison(
    model_scores: dict, output_path: str = "accuracy_comparison.png"
) -> None:
    """
    Create and save a bar chart comparing test accuracy across all
    trained machine learning models.

    Parameters
    ----------
    model_scores : dict
        Mapping of model name (str) to test accuracy (float, 0-1).
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating accuracy_comparison.png")

    model_names = list(model_scores.keys())
    accuracies = [score * 100 for score in model_scores.values()]

    plt.figure(figsize=(12, 7))
    colors = sns.color_palette("viridis", len(model_names))
    bars = plt.bar(model_names, accuracies, color=colors, edgecolor="black")

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{accuracy:.2f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.title("Model Accuracy Comparison", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Machine Learning Model", fontsize=13)
    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.ylim(0, 110)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list,
    model_name: str,
    output_path: str = "confusion_matrix.png",
) -> None:
    """
    Create and save a professional, annotated confusion matrix heatmap
    for the best-performing model.

    Parameters
    ----------
    y_true : array-like
        Ground-truth test labels.
    y_pred : array-like
        Predicted labels from the model.
    class_names : list
        Human-readable class names in label order.
    model_name : str
        Name of the model being evaluated (used in the plot title).
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating confusion_matrix.png")

    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Number of Samples"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Predicted Species", fontsize=13)
    plt.ylabel("Actual Species", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    feature_names: list,
    importances,
    model_name: str,
    output_path: str = "feature_importance.png",
) -> None:
    """
    Create and save a horizontal bar chart of feature importances for
    tree-based models (Decision Tree / Random Forest).

    Parameters
    ----------
    feature_names : list
        Names of the input features.
    importances : array-like
        Importance scores corresponding to each feature.
    model_name : str
        Name of the model being evaluated (used in the plot title).
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
    """
    print_subsection("Generating feature_importance.png")

    importance_series = pd.Series(importances, index=feature_names).sort_values()

    plt.figure(figsize=(11, 7))
    colors = sns.color_palette("crest", len(importance_series))
    plt.barh(importance_series.index, importance_series.values, color=colors, edgecolor="black")

    for index, value in enumerate(importance_series.values):
        plt.text(value + 0.005, index, f"{value:.3f}", va="center", fontsize=11, fontweight="bold")

    plt.title(f"Feature Importance — {model_name}", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Importance Score", fontsize=13)
    plt.ylabel("Feature", fontsize=13)
    plt.xlim(0, max(importance_series.values) + 0.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()