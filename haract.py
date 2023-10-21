#============================= IMPORT LIBRARIES =============================

import pandas as pd
from sklearn import preprocessing
from tkinter.filedialog import askopenfilename
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

#============================= DATA SELECTION ==============================

filename = askopenfilename()
dataframe=pd.read_csv(filename)
print(" -----------  Input Data  --------------- ")
print()
print(dataframe.head(20))
dataframe1=dataframe['Activity']


#============================= PREPROCESSING ==============================

#==== checking missing values ====

print(" -------------  Checking Missing Values  ---------------")
print()
print(dataframe.isnull().sum())


#==== label encoding ====

print("==================== Before Label Encoding   ===================")
print()
print(dataframe['Activity'].head(15))
print()
label_encoder=preprocessing.LabelEncoder()

print("==================== After Label Encoding   ===================")
print()
dataframe['Activity']=label_encoder.fit_transform(dataframe['Activity'])
print(dataframe['Activity'].head(15))
print()

#========================= FEATURE SELECTION ==============================

#=== split the data ====

x=dataframe.drop('Activity',axis=1)
x=abs(x)
y=dataframe['Activity']


from sklearn.feature_selection import SelectKBest, chi2
chi2_features = SelectKBest(chi2, k = 16) 
x_kbest= chi2_features.fit_transform(x, y)

print("============  faeture selection (Chi square)  ===================")
print()
print("The original features is : ", x.shape[1])
print()
print("The reduced features is : ", x_kbest.shape[1])
print()


#========================= DATA SPLITTING ==============================

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x_kbest,y,test_size = 0.03)

print("============  Data splitting  ===================")
print()
print("Total data    :",dataframe.shape[0])
print()
print("Training data :",X_train.shape[0])
print()
print("Testing data :",X_test.shape[0])
print()

#========================= CLASSIFICATION ==============================

#==== KNN ====

from sklearn.neighbors import KNeighborsClassifier

#initialize the model
knn = KNeighborsClassifier()

#fitting the model
knn = knn.fit(X_train,y_train)

#predict the model
y_pred_knn = knn.predict(X_test)

result_knn = (metrics.accuracy_score(y_pred_knn,y_test)) * 100

print("============  K Nearest Neighbour  ===================")
print()
print(" Accuracy for KNN :",result_knn,'%')
print()
print(metrics.classification_report(y_pred_knn,y_test))
print()

#========================= CLASSIFICATION ==============================
print("================= SVM===================")
from sklearn.svm import SVC
import seaborn as sns
model=SVC(kernel='linear', random_state=0)
model.fit(X_train,y_train)

x_pred=model.predict(X_test)

print("mean squared error=",metrics.mean_squared_error(y_test,x_pred)*100)
from sklearn.metrics import accuracy_score
svres=accuracy_score(y_test,x_pred)*100

import joblib
# Save the model and accuracy to a file
joblib.dump(knn, 'svm_model.pkl')
with open('svm_accuracy.txt', 'w') as f:
    f.write(f'{svres}')

print("SVM of Accuracy",svres)

#sns.countplot(y=x_pred)
#plt.show()

from sklearn.model_selection import GridSearchCV

# Define a grid of hyperparameters to search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(random_state=0), param_grid, cv=3)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model for predictions
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test)

# Evaluate the best model
best_accuracy = accuracy_score(y_test, y_pred_best)*100
print("Best Model Accuracy:", best_accuracy)

#=== recognize the activities =====

result=int(input("Enter the value : "))
print()
print("The predicted result is.....",result)


if y_pred_knn[result]== 0:
    print("============================")
    print()
    print(' Laying  ')
    print()
    print("============================")
elif y_pred_knn[result]==1:
    print("============================")
    print()
    print('Sitting ')
    print()
    print("============================")
elif y_pred_knn[result]==2:
    print("============================")
    print()
    print('Standing ')
    print()
    print("============================")
elif y_pred_knn[result]==3:
    print("============================")
    print()
    print('Walking ')
    print()
    print("============================")
elif y_pred_knn[result]==4:
    print("============================")
    print()
    print('Walking Downstairs ')
    print()
    print("============================")
elif y_pred_knn[result]==5:
    print("============================")
    print()
    print('Walking Upstairs ')
    print()
    print("============================")


#========================= VISUALIZATION ==============================

import numpy as np

import matplotlib.pyplot as plt


#count of each activity
count_of_each_activity = np.array(y_train.value_counts())
activities = sorted(dataframe1.unique())
print("=========== COUNT OF EACH ACTIVITY ===========")
plt.rcParams.update({'figure.figsize':[5,5],'font.size':12})
plt.pie(count_of_each_activity,labels=activities,autopct = '%0.2f')
plt.show()

vals=[svres,best_accuracy,result_knn]
inds=range(len(vals))
labels=["SVM","Hyperparameter Turning","KNN"]
fig,ax=plt.subplots()
reacts=ax.bar(inds,vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title("Comparison Graph - Accuracy")
plt.show()

