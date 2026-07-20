# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create images folder
os.makedirs("images", exist_ok=True)

# Load Iris Dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
df["Species"] = df["Species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("First 5 Rows of Dataset:\n")
print(df.head())

# -------------------------------
# Data Visualization
# -------------------------------

# Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Species", data=df)
plt.title("Class Distribution")
plt.savefig("images/class_distribution.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# -------------------------------
# Prepare Data
# -------------------------------

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Train Model
# -------------------------------

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.show()

# -------------------------------
# Sample Prediction
# -------------------------------

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nSample Flower Prediction:")

print("Input:", sample[0])

print("Predicted Species:", iris.target_names[prediction[0]])