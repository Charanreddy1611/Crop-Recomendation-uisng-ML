# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:08:00 2022

@author: chara
"""

#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')



#Importing Dataset
data = pd.read_csv(r"C:\Users\chara\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\sem 4\ml proj\Crop_recommendation.csv")

data.head()
print(data.head())
print('      ')

data.sample()
print(data.sample())
print('      ')

data.shape
print(data)
print('      ')

#Columns Description
'''*N - contains ratio of Nitrogen content in soil
P - contains ratio of Phosphorous content in soil
K - contains ratio of Potassium content in soil
Temperature - temperature in degree Celsius
Humidity - relative humidity in %
ph - ph value of the soil
Rainfall - rainfall in mm
Label - Predicted Crop'''

data.columns
print(data.columns)
print('     ')

data.dtypes
print(data.dtypes)
print('     ')

#Value_counts
data.label.value_counts()
print(data.label.value_counts())
print('     ')

data.isna().sum()
print(data.isna().sum())
print('     ')

data = data.rename(columns={'N': 'Nitrogen', 'P': 'Phosphorous', 'K': 'Potassium', 'label': 'Recommended_Crop'})
print(data.columns)
print('     ')

data.columns
print(data)
print('     ')

#Exploratory Data Analysis
# How Nitrogen are distributed
plt.figure(figsize=(11, 9), dpi=200)
sns.distplot(data['Nitrogen'], bins=20)
plt.show()

# How Phosphorous are distributed
plt.figure(figsize=(11, 9))
sns.distplot(data['Phosphorous'], bins=20)
plt.show()

# How Potassium are distributed
plt.figure(figsize=(11, 9))
sns.distplot(data['Potassium'], bins=20)
plt.show()

#different crops taken to train model
plt.figure(figsize=(17, 9))
rest = data['Recommended_Crop'].value_counts()[:20]
sns.barplot(rest, rest.index)
plt.title("Crops taken to train model")
plt.xlabel("Count")
plt.ylabel("Crops")
plt.show()

# rainfall is required for a crop
plt.figure(figsize=(17, 9))
ax = plt.subplot()
plt.xticks(rotation=90)
ax.bar(data["Recommended_Crop"], data["rainfall"])
plt.title('Rainfall Vs Crops')

#Pairplots and Heatmap
#Pairplot
sns.pairplot(data)

#Heatmap
data.corr()
sns.heatmap(data.corr(), annot=True, cmap="BuPu")

#Data Preprocessing
data.isna().sum()

# I)Outliers Finding and removing
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]

data.isnull().sum()

data.mean()

data.fillna(data.mean(), inplace=True)

data.isnull().sum()

# II)Splitting dataset into independent and dependent columns
x = data[['Nitrogen', 'Phosphorous', 'Potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['Recommended_Crop']
labels = data['Recommended_Crop']

x.head()

y.head()

# III)Splitting into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=99, stratify=y)

print('Test & Train Cases :')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#Applying Machine Learning Models
accuracy = []
models = []

# I)Logistic Regression
from sklearn.linear_model import LogisticRegression

cls = LogisticRegression()
cls.fit(x_train, y_train)

y_logistic = cls.predict(x_test)

# II)Decision Tree
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_logistic)
accuracy.append(acc)
models.append('Logistic Regression')
print('Accuracy of Logistic Regression:', acc)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_logistic))

from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier(criterion='entropy', random_state=0)
dec.fit(x_train, y_train)

y_dec = dec.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_dec)

train_accuracy = []
test_accuracy = []
plt.figure()
corr = data.corr()
sns.heatmap(corr, annot=True, cmap="BuPu")
plt.show()
for depth in range(1, 17):
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=10)
    dt_model.fit(x_train, y_train)
    train_accuracy.append(dt_model.score(x_train, y_train))
    test_accuracy.append(dt_model.score(x_test, y_test))

score = pd.DataFrame({'max_depth': range(1, 17), 'train_acc': train_accuracy, 'test_acc': test_accuracy})
score

plt.figure(figsize=(12, 6))
plt.plot(score['max_depth'], score['train_acc'], marker='*')
plt.plot(score['max_depth'], score['test_acc'], marker='+')
plt.title('train vs test based on max_depth')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
plt.show()

dt_model = DecisionTreeClassifier(max_depth=16, random_state=10)
dt_model.fit(x_train, y_train)

dt_model.score(x_train, y_train)

acc = dt_model.score(x_test, y_test)
accuracy.append(acc)
models.append('Decision Tree')
print('      ')
print('Accuracy Of Decision Tree:',acc)

from sklearn import tree

plt.figure(figsize=(35, 25))
tree.plot_tree(dt_model, filled=True, max_depth=3, feature_names=x_train.columns)
plt.show()

# III)KNN
from sklearn.neighbors import KNeighborsClassifier as KNN

cls = KNN()
cls.fit(x_train, y_train)

predict = cls.predict(x_test)

k = accuracy_score(predict, y_test)
print('      ')
print("Accuracy Score Of KNN :", k)


def elbow(k):
    all_error = []
    for i in k:
        cls = KNN(n_neighbors=i, p=2)  # Euclidean Dist
        cls.fit(x_train, y_train)
        predict = cls.predict(x_test)
        k = accuracy_score(predict, y_test)
        error = 1 - k
        all_error.append(error)
    return all_error


r = range(1, 10, 1)
test = elbow(r)

plt.plot(r, test)
plt.xlabel('K neighbours')
plt.ylabel('Test errors')
plt.title('Elbow curve')

cls = KNN(n_neighbors=3, p=2)  # Euclidean Dist
cls.fit(x_train, y_train)

predict = cls.predict(x_test)

k = accuracy_score(predict, y_test)
accuracy.append(k)
models.append('KNN')
print("Accuracy score  Of KNN:", k)

# IV)Knaive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_gnb = gnb.predict(x_test)

k = accuracy_score(y_gnb, y_test)
accuracy.append(k)

print('      ')
models.append('Naive Bayes')
print("accuracy score of Naive Bayes:", k)

# V)Random Forest
from sklearn.metrics import classification_report

print(classification_report(y_test, y_gnb))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

y_RFC = rfc.predict(x_test)

accuracy_score(y_test, y_RFC)

y_RFC = rfc.predict(x_test)

k = accuracy_score(y_RFC, y_test)
accuracy.append(k)
models.append('Random Forest')
print('      ')
print("Accuracy Score of Random Forest:", k)

# VI)Support Vector machine
from sklearn.svm import SVC

clf = SVC(kernel='rbf', random_state=0)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)

from sklearn.svm import SVC

clf = SVC(kernel='linear', random_state=0)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

k = accuracy_score(y_pred, y_test)
accuracy.append(k)
models.append('SVM')
print('      ')
print("Accuracy Score of SVM:", k)

#Comparision
plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=accuracy, y=models, palette='dark')

#Prediction
data = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])
prediction = rfc.predict(data)
print('      ')
print('Predicted Crop 1 : ',prediction)

data = np.array([[32, 137, 204, 22.860066, 93.128599, 5.824152, 117.729673]])
prediction = rfc.predict(data)
print('      ')
print('Predicted Crop 2 : ',prediction)