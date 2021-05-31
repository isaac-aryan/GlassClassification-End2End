#Imported Stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

#Reading Dataset
df = pd.read_csv('glass.csv')
print(df.head())

#Feature Information
#RI: refractive index
#Na: Sodium 
#Mg: Magnesium
#Al: Aluminum
#Si: Silicon
#K: Potassium
#Ca: Calcium
#Ba: Barium
#Fe: Iron

# 1 buildingwindowsfloatprocessed 
# 2 buildingwindowsnonfloatprocessed 
# 3 vehiclewindowsfloatprocessed 
# 4 vehiclewindowsnonfloatprocessed 
# 5 containers 
# 6 tableware 
# 7 headlamps
print(df.Ba.unique())
print(df.shape)

#Checking for null values
print(df.isnull().sum())

print(df.describe())

print(df.corr())
sns.pairplot(df)
plt.show()

#Correlations of each feature in dataset
corrmat = df.corr()
top_features = corrmat.index
plt.figure(figsize = (20,20))

g = sns.heatmap(df[top_features].corr(), annot = True, cmap = "Blues")
plt.show()

plt.figure()
df.hist(figsize=(20,20))
plt.show()

#Setting independant and target variables
X = df.drop('Type',axis=1)
y = df['Type']

print(X.head())
print(y.head())

#Setting and fitting model
model = XGBClassifier()
model.fit(X, y)

#Feature importances and visualising it
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index = X.columns)
feat_importances.nlargest(5).plot(kind = 'barh')
plt.show()

#Splitting Data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True)
xgboostcl = XGBClassifier()
xgboostcl.fit(X_train, y_train)

#Predictions
preds = xgboostcl.predict(X_test)
print("Predictions:\n", preds)
print("\nTest Values:\n", y_test.values)
print("\nAccuracy:", accuracy_score(preds,y_test.values))

#Pickling and dumping
file = open('xgbcl_model.pkl', 'wb')
pickle.dump(xgboostcl, file)

#checking git

