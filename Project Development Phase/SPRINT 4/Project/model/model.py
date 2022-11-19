import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score,train_test_split
from sklearn import linear_model, tree, ensemble
from sklearn.model_selection import GridSearchCV
import pickle

data1=pd.read_csv("../dataset/Heart_Disease_Prediction.csv")
data1.head()
data1.isnull().sum()
data1.describe()
X = data1.drop(['target'],axis='columns')
y = data1.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

model_4 = tree.DecisionTreeClassifier(criterion='entropy')
model_4.fit(X_train, y_train)

# dt=model_4.score(X_train, y_train)

pickle.dump(model_4, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl','rb'))
model.predict([[56,1,1,120,236,1,1,178,0,0.8,2,0,2]])
