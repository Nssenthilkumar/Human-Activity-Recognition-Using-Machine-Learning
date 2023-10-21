import pandas as pd
from sklearn import preprocessing
from tkinter.filedialog import askopenfilename
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
import joblib
from sklearn.metrics import accuracy_score

filename = askopenfilename()
dataframe=pd.read_csv(filename)
print(" -----------  Input Data  --------------- ")
print()
print(dataframe.head(20))
dataframe1=dataframe['Activity']

print(" -------------  Checking Missing Values  ---------------")
print()
print("to check the datatset\n",dataframe.isnull())
print("\nisnull.sum() function, to assign the 0 or 1\n",dataframe.isnull().sum())

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

from sklearn.feature_selection import SelectKBest, chi2
chi2_features = SelectKBest(chi2, k = 16) 
x_kbest= chi2_features.fit_transform(x, y)

print("============  faeture selection (Chi square)  ===================")
print()
print("The original features is : ", x.shape[1])
print()
print("The reduced features is : ", x_kbest.shape[1])
print()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x_kbest,y,test_size = 0.03,random_state = 0)

print("============  Data splitting  ===================")
print()
print("Total data    :",dataframe.shape[0])
print()
print("Training data :",X_train.shape[0])
print()
print("Testing data :",X_test.shape[0])
print()

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
joblib.dump(model, 'svm_model.pkl')
with open('svm_accuracy.txt', 'w') as f:
    f.write(f'{svres}')


print("SVM of Accuracy",svres)

#sns.countplot(y=x_pred)
#plt.show()

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
# Define a grid of hyperparameters to search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

# Create a GridSearchCV object
grid_search = RandomizedSearchCV(SVC(random_state=0), param_grid, cv=5)

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

