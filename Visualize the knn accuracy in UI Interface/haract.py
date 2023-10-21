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

from sklearn.neighbors import KNeighborsClassifier

#initialize the model
knn = KNeighborsClassifier()

#fitting the model
knn = knn.fit(X_train,y_train)

#predict the model
y_pred_knn = knn.predict(X_test)

accuracy= (metrics.accuracy_score(y_pred_knn,y_test)) * 100

joblib.dump(knn, 'knn_model.pkl')
with open('knn_accuracy.txt', 'w') as f:
    f.write(f'{accuracy}')

print("============  K Nearest Neighbour  ===================")
print()
print(" Accuracy for KNN :",accuracy,'%')
print()
print(metrics.classification_report(y_pred_knn,y_test))
print()

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

import numpy as np

import matplotlib.pyplot as plt


#count of each activity
count_of_each_activity = np.array(y_train.value_counts())
activities = sorted(dataframe1.unique())
print("=========== COUNT OF EACH ACTIVITY ===========")
plt.rcParams.update({'figure.figsize':[5,5],'font.size':12})
plt.pie(count_of_each_activity,labels=activities,autopct = '%0.2f')
plt.show()
