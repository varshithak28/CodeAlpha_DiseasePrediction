import pandas as pd
data = pd.read_csv("heart.csv")
print(data.head())
print(data.shape)
print(data.columns)
print(data.info())

#Logistic regression code
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Features and target
X = data.drop("target", axis=1)
y = data["target"]

#Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Create model
model = LogisticRegression(max_iter=1000)

#Train model
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

#Graph
plt.bar(["Accuracy"], [accuracy])
plt.ylabel("Score")
plt.title("Disease Prediction Accuracy")
plt.show()