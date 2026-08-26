# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries.

2. Load the spam mail dataset (spam.csv).

3. Select the relevant columns (message and label).

4. Encode the class labels (spam = 1, ham = 0).

5. Visualize the distribution of spam and ham emails.

6. Separate features (email text) and target labels.

7. Split the dataset into training and testing sets.

8. Convert text data into numerical features using TF-IDF.

9. Initialize the Support Vector Machine classifier.

10. Define hyperparameters and perform Grid Search for tuning.

11. Train the SVM model using the best parameters.

12. Predict the class labels for test data.

13. Evaluate the model using accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ASWIN B
RegisterNumber:  212224110007
*/
```
```python
# Exp 11 – SVM for Spam Mail Detection with Visualization

# 1. Import Required Libraries
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Detect File Encoding
file_path = "spam.csv"   # use "/content/spam.csv" for Colab

with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print("Detected Encoding:", result)

# 3. Load Dataset
data = pd.read_csv(file_path, encoding=result['encoding'])

# 4. Basic Data Exploration
print(data.head())
print(data.info())
print(data.isnull().sum())

# 5. Visualization: Spam vs Ham Distribution
plt.figure(figsize=(5,4))
sns.countplot(x=data['v1'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# 6. Message Length Visualization
data['msg_length'] = data['v2'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(data=data, x='msg_length', hue='v1', bins=50, kde=True)
plt.title("Message Length Distribution (Spam vs Ham)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# 7. Feature and Target Selection
x = data['v2'].values     # messages
y = data['v1'].values     # labels

# 8. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# 9. Text Vectorization (Bag of Words)
cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

# 10. Initialize and Train SVM
svc = SVC(kernel='linear')
svc.fit(x_train_vec, y_train)

# 11. Prediction
y_pred = svc.predict(x_test_vec)

# 12. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 13. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix – SVM Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

```

## Output:

<img width="1017" height="570" alt="image" src="https://github.com/user-attachments/assets/07df13b6-3ab1-4207-a71c-d6d9bed226b2" />

<img width="472" height="391" alt="image" src="https://github.com/user-attachments/assets/fd8000c5-9de4-4441-afd0-3068a791a8ab" />

<img width="549" height="391" alt="image" src="https://github.com/user-attachments/assets/8250a2a8-9f0c-4f7d-8edf-471c92457740" />

<img width="537" height="195" alt="image" src="https://github.com/user-attachments/assets/883c5993-6715-4f17-87d1-9d981115a505" />

<img width="444" height="391" alt="image" src="https://github.com/user-attachments/assets/c1c867b6-0f54-4389-bc2a-3c873971045d" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
