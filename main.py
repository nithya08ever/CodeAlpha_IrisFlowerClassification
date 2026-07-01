# Import required libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("dataset/Iris.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# Remove Id column if present
# ==========================

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# ==========================
# Data Visualization
# ==========================

sns.pairplot(df, hue="Species")
plt.savefig("screenshots/pairplot.png")
plt.close()

# ==========================
# Feature Selection
# ==========================

X = df.drop("Species", axis=1)

y = df["Species"]

# Convert species into numbers

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Accuracy
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy : ", accuracy * 100)

# ==========================
# Classification Report
# ==========================

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("screenshots/confusion_matrix.png")

plt.show()

# ==========================
# Sample Prediction
# ==========================

sample = pd.DataFrame({
    "SepalLengthCm": [5.1],
    "SepalWidthCm": [3.5],
    "PetalLengthCm": [1.4],
    "PetalWidthCm": [0.2]
})

prediction = model.predict(sample)

print("\nSample Prediction:")

print(encoder.inverse_transform(prediction))
print("\nProject Completed Successfully!")