pip install ucimlrepo

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix, silhouette_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

from ucimlrepo import fetch_ucirepo


# fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# data (as pandas dataframes)
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

frames = [X, y]

df = pd.concat(frames,axis=1)

df.head()

#Renaming the columns
#1-5 (Ritish)
new_column_names = {
    'X1': 'amt_credit',
    'X2': 'gender',
    'X3': 'education',
    'X4': 'marital_status',
    'X5': 'age',
#6-11 (Abhishek)
    'X6': 'pay_sep',
    'X7': 'pay_aug',
    'X8': 'pay_jul',
    'X9': 'pay_jun',
    'X10': 'pay_may',
    'X11':'pay_apr',
    #MonthlyPaymentRecord

#12-17 (Shashank) #bill_sept = September bill amount
    "X12":"bill_sep",
    "X13":"bill_aug",
    "X14":"bill_jul",
    "X15":"bill_jun",
    "X16":"bill_may",
    "X17":"bill_apr",

#Tommy 18-23
    'X18': 'pre_sep',
    'X19': 'pre_aug',
    'X20': 'pre_jul',
    'X21': 'pre_jun',
    'X22': 'pre_may',
    'X23': 'pre_apr',
    'Y': 'willDefault'
    #Amount of previous payment (NT dollar)
}

df.rename(columns=new_column_names, inplace=True)

print(df.head())



#Replacing all 0's,5's and 6's with 4
df['education']=df['education'].replace(0,4).replace(5,4).replace(6,4)

#Marital Status
df['marital_status']=df['marital_status'].replace(0,3)

#Repayment Status
test_columns=['pay_sep', 'pay_aug', 'pay_jul', 'pay_jun','pay_may','pay_apr']
df[test_columns]=df[test_columns].replace(-2,0).replace(-1,0)

import matplotlib.pyplot as plt
df_plot=df.copy()
df_plot["gender"]=df["gender"].map({1:"Male",2:"Female"})
grouped_data = df_plot.groupby('gender')['willDefault'].mean() * 100
grouped_data.plot(kind='bar', color=['C0', 'C1'])
plt.xlabel('Gender')
plt.ylabel('Percentage of People Defaulting')
plt.title("Gender vs Percentage of People Defaulting in that particular class ")


import matplotlib.pyplot as plt
df_plot=df.copy()
df_plot["education"]=df["education"].map({1:"Graduate School",2:"University",3:"High School",4:"Others"})
reorder_order = ['High School', 'University', 'Graduate School', 'Others']
grouped_data = df_plot.groupby('education')['willDefault'].mean() * 100
grouped_data=grouped_data.reindex(index=reorder_order)
grouped_data.plot(kind='bar', color=['C0', 'C1','C2','C3'])
plt.xlabel('Education Level')
plt.ylabel('Percentage of People Defaulting')
plt.title("Level of Education vs Percentage of People Defaulting in that particular class")

import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(df['amt_credit'], bins=25,kde=True)
plt.xlabel('Credit Limit (In NT Dollars)')
plt.ylabel('Frequency')
plt.title('Distribution of Credit Limit')
plt.grid(True)
plt.show()

print("Minimum Age:",df['age'].min())
print("Maximunm Age:",df['age'].max())

df_plot = df.copy()

df_plot['ageClass'] = ""

for i in df_plot.index:
    age = df_plot.at[i, 'age']
    if (21 <= age <= 30):
        df_plot.at[i, 'ageClass'] = "Young Adults"
    elif (31 <= age <= 45):
        df_plot.at[i, 'ageClass'] = "Middle Aged Adults"
    elif (46 <= age <= 60):
        df_plot.at[i, 'ageClass'] = "Old Adults"
    else:
        df_plot.at[i, 'ageClass'] = "Seniors"


import matplotlib.pyplot as plt

reorder_order = ['Young Adults', 'Middle Aged Adults', 'Old Adults', 'Seniors']

grouped_data = df_plot.groupby('ageClass')['willDefault'].mean() * 100

grouped_data=grouped_data.reindex(index=reorder_order)
grouped_data.plot(kind='bar', color=['C0', 'C1','C2','C3'])
plt.xlabel('Age Group')
plt.ylabel('Percentage of People Defaulting')
plt.title("Age Group vs Percentage of People Defaulting ")

df['amt_credit'].describe()

df_plot=df.copy()
Q1 = df_plot['amt_credit'].quantile(0.25)
Q2 = df_plot['amt_credit'].quantile(0.75)
Q3 = df_plot['amt_credit'].quantile(0.9)

low_limit = Q1
medium_limit = Q2
high_limit = Q3
very_high_limit = df['amt_credit'].max()

#print(Q1,Q2,Q3,very_high_limit)

def classify_credit_limit(limit):
    if limit <= low_limit:
        return 'Low Credit Limit'
    elif low_limit < limit <= medium_limit:
        return 'Medium Credit Limit'
    elif medium_limit < limit <= high_limit:
        return 'High Credit Limit'
    elif high_limit < limit <= very_high_limit:
        return 'Very High Credit Limit'

df_plot['credit_limit_category'] = df_plot['amt_credit'].apply(classify_credit_limit)

#df_plot['credit_limit_category'].head()

import matplotlib.pyplot as plt

reorder_order = ['Low Credit Limit', 'Medium Credit Limit', 'High Credit Limit', 'Very High Credit Limit']

grouped_data = df_plot.groupby('credit_limit_category')['willDefault'].mean() * 100
grouped_data=grouped_data.reindex(index=reorder_order)
grouped_data.plot(kind='bar', color=['C0', 'C1','C2','C3'])
plt.xlabel('Credit Limit Category')
plt.ylabel('Percentage of People Defaulting')
plt.title("Credit Limit Category vs Percentage of People Defaulting in that particular class")

df_plot=df.copy()
df_plot["willDefault"]=df["willDefault"].map({0:"Will Not Default",1:"Will Default"})

df_plot['willDefault'].value_counts().plot(kind='bar',color=['C0', 'C1'])
plt.xlabel("Default on the payment")
plt.ylabel("Count")
plt.title("Number of people v/s Defaulting status")

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(),cmap='coolwarm', fmt=".2f")
plt.show()

X = df.drop(['willDefault'], axis=1)
y = df['willDefault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("\nShape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)

pipeline = Pipeline(steps=[
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train,y_train)

#Predicting on Test Data
y_pred = pipeline.predict(X_test)  # testing
y_prob = pipeline.predict_proba(X_test)[:, 1]

#Compute and plot the ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6)) # creating a figure of size (8,6)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})') # plotting ROC
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RandomForest Classifier')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score (calculated with y_pred): {roc_auc}")
print(f"ROC-AUC Score (calculated with y_prob): {roc_auc_value}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Reset the accuracy list in case this is being rerun
accuracy_list = []
precision_list = []
recall_list = []

# Ensure the loop covers the range from 1 to 50
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  # Use raw data for training
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    # Print statement to confirm each K value is processed
    #print(f'K={k}, Accuracy={accuracy:.4f}')
    print(f'K={k},Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}')

# from sklearn.metrics import precision_score, recall_score

# precision_list = []
# recall_list = []

# # Loop over K values
# for k in range(1, 51):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)

#     precision = precision_score(y_test, y_pred, average='macro')
#     recall = recall_score(y_test, y_pred, average='macro')

#     precision_list.append(precision)
#     recall_list.append(recall)

#     print(f'K={k}, Precision={precision:.4f}, Recall={recall:.4f}')


plt.figure(figsize=(15, 8))
plt.plot(range(1, 51), precision_list, label='Precision', marker='o', linestyle='dashed', color='blue')
plt.plot(range(1, 51), recall_list, label='Recall', marker='o', linestyle='dashed', color='green')
plt.plot(range(1, 51), accuracy_list, label='Accuracy', marker='o', linestyle='dashed', color='red')
plt.title('Accuracy, Precision, and Recall vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Score')
plt.xticks(range(1, 51, 2))
plt.legend()
plt.grid(True)
plt.show()


# Finding the best K value for accuracy
best_k_accuracy = accuracy_list.index(max(accuracy_list)) + 1
best_accuracy = max(accuracy_list)
print(f'The best K value for accuracy is {best_k_accuracy} with an accuracy of {best_accuracy:.4f}')

# Finding the best K value for precision
best_k_precision = precision_list.index(max(precision_list)) + 1
best_precision = max(precision_list)
print(f'The best K value for precision is {best_k_precision} with a precision of {best_precision:.4f}')

# Finding the best K value for recall
best_k_recall = recall_list.index(max(recall_list)) + 1
best_recall = max(recall_list)
print(f'The best K value for recall is {best_k_recall} with a recall of {best_recall:.4f}')


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Ensure you're using the raw data split
knn = KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train, y_train)  # Use the raw training data

# Predict probabilities for the test set
y_prob = knn.predict_proba(X_test)[:, 1]  # Assuming binary classification

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN Classifier (K=42)')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Fit the KNN Classifier with K=42, using the raw data
knn = KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]  # Assuming binary classification

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc_pred = roc_auc_score(y_test, y_pred)  # ROC-AUC based on the predictions
roc_auc_prob = roc_auc_score(y_test, y_prob)  # ROC-AUC based on the probability estimates

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score(y_pred): {roc_auc_pred:.4f}")
print(f"ROC-AUC Score (y_prob): {roc_auc_prob:.4f}")


knn = KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train, y_train)  # Fit using the raw training data
y_pred = knn.predict(X_test)  # Predictions on the raw test data
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for KNN Classifier (K=42) - Raw Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



X = df.drop(['willDefault'], axis=1)
y = df['willDefault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ('model', LogisticRegression(max_iter=500))
])

pipeline.fit(X_train,y_train)

#Predicting on Test Data
y_pred = pipeline.predict(X_test)  # testing
y_prob = pipeline.predict_proba(X_test)[:, 1]

#Compute and plot the ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6)) # creating a figure of size (8,6)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})') # plotting ROC
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score(y_pred): {roc_auc}")
print(f"ROC-AUC Score (y_prob): {roc_auc_value}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



X = df.drop(['willDefault'], axis=1)
y = df['willDefault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import GaussianNB
pipeline = Pipeline(steps=[
    ('model', GaussianNB())
])

pipeline.fit(X_train,y_train)

#Predicting on Test Data
y_pred = pipeline.predict(X_test)  # testing
y_prob = pipeline.predict_proba(X_test)[:, 1]

#Compute and plot the ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6)) # creating a figure of size (8,6)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})') # plotting ROC
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naive Bayes')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score(y_pred): {roc_auc}")
print(f"ROC-AUC Score (y_prob): {roc_auc_value}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# Resample the dataset
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print the class counts after resampling
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("\nShape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)

pipeline = Pipeline(steps=[
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RandomForest Classifier (Balanced Data)')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score (calculated with y_pred): {roc_auc}")
print(f"ROC-AUC Score (calculated with y_prob): {roc_auc_value}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score



from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)

# Resample the dataset
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print the class counts after resampling
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

# Reset the accuracy list in case this is being rerun
accuracy_list = []
precision_list = []
recall_list = []

# Ensure the loop covers the range from 1 to 50
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  # Use raw data for training
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    # Print statement to confirm each K value is processed
    #print(f'K={k}, Accuracy={accuracy:.4f}')
    print(f'K={k},Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}')

# # Initialize lists to store precision and recall values
# precision_list = []
# recall_list = []

# # Loop over K values
# for k in range(1, 51):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_resampled, y_resampled)  # Use the correct variable names for your resampled data
#     y_pred = knn.predict(X_test)

#     # Calculate precision and recall
#     precision = precision_score(y_test, y_pred, average='macro')
#     recall = recall_score(y_test, y_pred, average='macro')

#     precision_list.append(precision)
#     recall_list.append(recall)

#     print(f'K={k}, Precision={precision:.4f}, Recall={recall:.4f}')


plt.figure(figsize=(15, 8))
# Plot accuracy
plt.plot(range(1, 51), accuracy_list, label='Accuracy', marker='o', linestyle='dashed', markersize=5, color='red')
# Plot precision
plt.plot(range(1, 51), precision_list, label='Precision', marker='o', linestyle='dashed', markersize=5, color='blue')
# Plot recall
plt.plot(range(1, 51), recall_list, label='Recall', marker='o', linestyle='dashed', markersize=5, color='green')

plt.title('Model Evaluation Metrics vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Score')
plt.xticks(range(1, 51, 2))
plt.legend()
plt.grid(True)
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming X_resampled and y_resampled are your training data after oversampling
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_resampled, y_resampled)
# y_prob should be the probability of the class of interest.
# If your positive class is the second one, you use [:, 1]
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN Classifier (K=1)')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score (calculated with y_pred): {roc_auc}")
print(f"ROC-AUC Score (calculated with y_prob): {roc_auc_value}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming KNN is already imported and fitted for K=1 with oversampled data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_resampled, y_resampled)

# Generate predictions
y_pred = knn.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for KNN Classifier (K=1) with Balanced Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resample the dataset
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print the class counts after resampling
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("\nShape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)

pipeline = Pipeline(steps=[
    ('model', LogisticRegression(max_iter=500))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Balanced Data)')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score(y_pred): {roc_auc}")
print(f"ROC-AUC Score (y_prob): {roc_auc_value}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)

# Resample the dataset
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print the class counts after resampling
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("\nShape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)

pipeline = Pipeline(steps=[
    ('model', GaussianNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naive Bayes (Balanced Data)')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score(y_pred): {roc_auc}")
print(f"ROC-AUC Score (y_prob): {roc_auc_value}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

from sklearn.cluster import KMeans

#Kmeans sources: https://www.w3schools.com/python/python_ml_k-means.asp , https://www.youtube.com/watch?v=iNlZ3IU5Ffw

kdata = df

inertias = []

for i in range (1,11):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(kdata)
  inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#K should equal 3 based on our elbow method

kmeans = KMeans(n_clusters=3)
kmeans.fit(kdata)
kdata["results"] = kmeans.labels_
kdata.head()

plt.scatter(x=kdata['amt_credit'],y=kdata['age'],c=kdata['results'])
plt.show()

for k in range(1,6):
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(kdata)
  kdata[f'KMeans_{k}'] = kmeans.labels_

test, axs = plt.subplots(nrows=1,ncols=5,figsize=(20,5))

for i, ax in enumerate(test.axes, start=1):
  ax.scatter(x=kdata['amt_credit'],y=kdata['age'],c=kdata[f'KMeans_{i}'])
  ax.set_title(f'N Clusters: {i}')

#Kmeans sources: https://www.w3schools.com/python/python_ml_k-means.asp , https://www.youtube.com/watch?v=iNlZ3IU5Ffw

kdata_c = df[["amt_credit","gender","education","marital_status","age","willDefault"]]

inertias = []

for i in range (1,11):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(kdata_c)
  inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#K should equal 3 based on our elbow method

kmeans = KMeans(n_clusters=3)
kmeans.fit(kdata_c)
kdata_c["KMeans_3"] = kmeans.labels_
kdata_c.head()

plt.scatter(x=kdata_c['amt_credit'],y=kdata_c['age'],c=kdata_c['KMeans_3'])
plt.show()

#plt.scatter(x=kdata_c['amt_credit'],y=kdata_c['marital_status'],c=kdata_c['KMeans_3'])
#plt.show()

#plt.scatter(x=kdata_c['willDefault'],y=kdata_c['education'],c=kdata_c['KMeans_3'])
#plt.show()

#plt.scatter(x=kdata_c['gender'],y=kdata_c['amt_credit'],c=kdata_c['KMeans_3'])

#plt.show()

#Ignores gender, age, marital status
#Maybe credit amount, education, default chance

group1 = kdata_c[kdata_c["KMeans_3"] == 0] #4563 people
group2 = kdata_c[kdata_c["KMeans_3"] == 1] #14541 people
group3 = kdata_c[kdata_c["KMeans_3"] == 2] #10896 people

plt.bar(["Group 1","Group 2","Group 3"],[group1['amt_credit'].mean(),group2['amt_credit'].mean(),group3['amt_credit'].mean()])
plt.show()

plt.bar(["Group 1","Group 2","Group 3"],[group1['age'].mean(),group2['age'].mean(),group3['age'].mean()])
plt.show()

print(group1['willDefault'].value_counts())
print(group2['willDefault'].value_counts())
print(group3['willDefault'].value_counts())


for k in range(1,6):
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(kdata_c)
  kdata_c[f'KMeans_{k}'] = kmeans.labels_

test, axs = plt.subplots(nrows=1,ncols=5,figsize=(20,5))

for i, ax in enumerate(test.axes, start=1):
  ax.scatter(x=kdata_c['amt_credit'],y=kdata_c['age'],c=kdata_c[f'KMeans_{i}'])
  ax.set_title(f'N Clusters: {i}')

#test2, axs = plt.subplots(nrows=1,ncols=5,figsize=(20,5))
#for i, ax in enumerate(test2.axes, start=1):
#  ax.scatter(x=kdata_c['amt_credit'],y=kdata_c['marital_status'],c=kdata_c[f'KMeans_{i}'])
#  ax.set_title(f'N Clusters: {i}')

#test3, axs = plt.subplots(nrows=1,ncols=5,figsize=(20,5))
#for i, ax in enumerate(test3.axes, start=1):
#  ax.scatter(x=kdata_c['willDefault'],y=kdata_c['education'],c=kdata_c[f'KMeans_{i}'])
#  ax.set_title(f'N Clusters: {i}')

#test4, axs = plt.subplots(nrows=1,ncols=5,figsize=(20,5))
#for i, ax in enumerate(test4.axes, start=1):
#  ax.scatter(x=kdata_c['gender'],y=kdata_c['amt_credit'],c=kdata_c[f'KMeans_{i}'])
#  ax.set_title(f'N Clusters: {i}')