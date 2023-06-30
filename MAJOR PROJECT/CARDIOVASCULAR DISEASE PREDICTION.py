#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DATA PRE-PROCESSING
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the data from a CSV file
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/cardio_train.csv', delimiter=';')

# Drop the 'id' column as it does not contribute to the prediction
data = data.drop('id', axis=1)

# Convert age from days to years
data['age'] = data['age'] // 365

# Convert gender to binary (0 for female, 1 for male)
data['gender'] = data['gender'].map({1: 0, 2: 1})

# Perform feature scaling on numerical attributes using StandardScaler
scaler = StandardScaler()
numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Convert categorical attributes to one-hot encoded representation
categorical_cols = ['cholesterol', 'gluc']
data = pd.get_dummies(data, columns=categorical_cols)

# Print the pre-processed data
print(data.head())


# In[2]:


#DATA VISUALIZATION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from a CSV file
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/cardio_train.csv', delimiter=';')

# Drop the 'id' column as it does not contribute to the analysis
data = data.drop('id', axis=1)

# Convert age from days to years
data['age'] = data['age'] // 365

# Convert gender to meaningful labels
data['gender'] = data['gender'].map({1: 'Female', 2: 'Male'})

# Convert cardiovascular disease label to meaningful labels
data['cardio'] = data['cardio'].map({0: 'No', 1: 'Yes'})

# Plotting count of individuals with and without cardiovascular disease
sns.countplot(data=data, x='cardio')
plt.title('Count of Individuals with and without Cardiovascular Disease')
plt.xlabel('Cardiovascular Disease')
plt.ylabel('Count')
plt.show()

# Plotting age distribution for individuals with and without cardiovascular disease
sns.histplot(data=data, x='age', hue='cardio', kde=True, multiple='stack')
plt.title('Age Distribution for Individuals with and without Cardiovascular Disease')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Plotting gender distribution for individuals with and without cardiovascular disease
sns.countplot(data=data, x='gender', hue='cardio')
plt.title('Gender Distribution for Individuals with and without Cardiovascular Disease')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()



# In[8]:


# Plotting correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the data from a CSV file
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/cardio_train.csv', delimiter=';')

# Drop the 'id' column as it does not contribute to the prediction
data = data.drop('id', axis=1)

# Split the dataset into features (X) and target variable (y)
X = data.drop('cardio', axis=1)
y = data['cardio']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifiers
svm = SVC()
svm.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions on the testing set
svm_preds = svm.predict(X_test)
knn_preds = knn.predict(X_test)
dt_preds = dt.predict(X_test)
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

# Calculate accuracy for each classifier
svm_accuracy = accuracy_score(y_test, svm_preds)
knn_accuracy = accuracy_score(y_test, knn_preds)
dt_accuracy = accuracy_score(y_test, dt_preds)
lr_accuracy = accuracy_score(y_test, lr_preds)
rf_accuracy = accuracy_score(y_test, rf_preds)

# Print the accuracy levels
print('Support Vector Machines (SVM) Accuracy:', svm_accuracy)
print('K-Nearest Neighbors (KNN) Accuracy:', knn_accuracy)
print('Decision Trees (DT) Accuracy:', dt_accuracy)
print('Logistic Regression (LR) Accuracy:', lr_accuracy)
print('Random Forest (RF) Accuracy:', rf_accuracy)


# In[11]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the data from a CSV file
data = pd.read_csv('C:/Users/akhil/OneDrive/Desktop/cardio_train.csv', delimiter=';')

# Drop the 'id' column as it does not contribute to the prediction
data = data.drop('id', axis=1)

# Split the dataset into features (X) and target variable (y)
X = data.drop('cardio', axis=1)
y = data['cardio']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Random Forest Accuracy:', accuracy)


# In[ ]:




