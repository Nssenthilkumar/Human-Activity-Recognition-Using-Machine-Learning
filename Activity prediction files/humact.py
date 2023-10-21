import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
import pickle
df=pd.read_csv("HARdataset.csv")
from sklearn import preprocessing
print(df.head())
print()

#========================= FEATURE SELECTION ==============================


X=df[["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z", "tBodyAcc-std()-X", "tBodyAcc-std()-Y", "tBodyAcc-std()-Z", "tBodyAcc-mad()-X", "tBodyAcc-mad()-Y", "tBodyAcc-mad()-Z", "tBodyAcc-max()-X", "tBodyAcc-max()-Y", "tBodyAcc-max()-Z", "tBodyAcc-min()-X", "tBodyAcc-min()-Y", "tBodyAcc-min()-Z","subject"]]
#X=df[["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z"]]
X=abs(X)
print(X)
y=df['Activity']
print("Activity",y)


from sklearn.feature_selection import SelectKBest, chi2
chi2_features = SelectKBest(chi2, k = 16) 
x_kbest= chi2_features.fit_transform(X, y)

print("============  faeture selection (Chi square)  ===================")
print()
print("The original features is : ", X.shape[1])
print()
print("The reduced features is : ", x_kbest.shape[1])

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x_kbest,y,test_size = 0.03,random_state = 0)


from sklearn.neighbors import KNeighborsClassifier

#initialize the model
knn = KNeighborsClassifier()

#fitting the model
knn = knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)

# Make pickle file of our model
pickle.dump(knn, open("model.pkl", "wb"))
