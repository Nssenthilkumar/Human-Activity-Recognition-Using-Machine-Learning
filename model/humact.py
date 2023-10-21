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

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

