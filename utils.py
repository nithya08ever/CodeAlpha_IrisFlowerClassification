import os


def print_section_header(title: str) -> None:
    """
    Print a nicely formatted section header to the terminal.

    This is used throughout the project to make console output easy
    to scan, so reviewers can quickly follow each stage of the pipeline.

    Parameters
    ----------
    title : str
        The heading text to display.

    Returns
    -------
    None
    """
    line_length = 70
    print("\n" + "=" * line_length)
    print(f"{title.upper():^{line_length}}")
    print("=" * line_length)


def print_subsection(title: str) -> None:
    """
    Print a smaller, secondary heading used to separate sub-steps
    inside a larger section.

    Parameters
    ----------
    title : str
        The sub-heading text to display.

    Returns
    -------
    None
    """
    print(f"\n--- {title} ---")


def ensure_directory_exists(directory_path: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory_path : str
        Path of the directory to check/create.

    Returns
    -------
    None
    """
    if directory_path and not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def format_percentage(value: float) -> str:
    """
    Convert a fractional score (e.g. 0.9667) into a readable
    percentage string (e.g. "96.67%").

    Parameters
    ----------
    value : float
        A numeric score between 0 and 1.

    Returns
    -------
    str
        The formatted percentage string.
    """
    return f"{value * 100:.2f}%"


# Centralized project constants so every script stays in sync.
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_FILENAME = "saved_model.pkl"
SCALER_FILENAME = "saved_scaler.pkl"

# Human-friendly species names indexed by the integer label used by
# scikit-learn's Iris dataset (0 = setosa, 1 = versicolor, 2 = virginica).
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]