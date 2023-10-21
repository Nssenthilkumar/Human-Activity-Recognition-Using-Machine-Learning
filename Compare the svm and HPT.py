import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

dataframe=pd.read_csv("HARdataset.csv")
print("----------------------------------------------------------")
print("Data Selection")
print()
print(dataframe.head(10))
print()
print("-----------------------------------------------------------")
list(dataframe.columns)

from sklearn import preprocessing
from sklearn import metrics

dataframe=pd.read_csv("HARdataset.csv")

print(" ------------  Input Data  --------------- ")
print()
print(dataframe.head(20))
dataframe1=dataframe['Activity']

print(" -------------  Checking Missing Values  ---------------")
print()
print(dataframe.isnull().sum())

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

x=dataframe.drop('Activity',axis=1)
x=abs(x)
y=dataframe['Activity']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 30)
from sklearn.preprocessing import StandardScaler
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

print("================= SVM===================")
from sklearn.svm import SVC
import seaborn as sns
model=SVC(kernel='linear', C=5.5, random_state=30)
model.fit(X_train,y_train)

x_pred=model.predict(X_test)

print("mean squared error=",metrics.mean_squared_error(y_test,x_pred)*100)
from sklearn.metrics import accuracy_score
svres=accuracy_score(y_test,x_pred)*100
print("SVM of Accuracy",svres)

sns.countplot(y=x_pred)
plt.show()

from sklearn.model_selection import GridSearchCV


param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}


grid_search = GridSearchCV(SVC(random_state=0), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)*100
print("Best Model Accuracy:", best_accuracy)

vals=[svres,best_accuracy]
inds=range(len(vals))
labels=["SVM","Hyperparameter turning"]
fig,ax=plt.subplots()
reacts=ax.bar(inds,vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title("Comparing the Accuracy Graph")
plt.show()

