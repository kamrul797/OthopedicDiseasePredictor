import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer as DV


train = pd.read_csv('trainset3.csv')
test = pd.read_csv('testset3.csv')

numeric_cols = [ 'id', 'age', 'salary' ]
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

cat_train = train.drop( numeric_cols + [ 'class'], axis = 1 )
cat_test = test.drop( numeric_cols + [ 'class'], axis = 1 )

x_cat_train = cat_train.to_dict( orient = 'records' )
x_cat_test = cat_test.to_dict( orient = 'records' )

vectorizer = DV( sparse = False )

vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

X_train = np.hstack(( x_num_train, vec_x_cat_train ))
X_test = np.hstack(( x_num_test, vec_x_cat_test ))
y_train = train['class']
y_test = test['class']

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train,y_train)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train.values.ravel())

svm =  SVC(kernel="linear", C=0.005)
svm.fit(X_train,y_train)

lrc = LogisticRegression()
lrc.fit(X_train,y_train)


p1 = dtc.predict(X_test)
p2 = knn.predict(X_test)
p3 = svm.predict(X_test)
p4 = lrc.predict(X_test)


accuracy1 = accuracy_score(y_test,p1)
accuracy2 = accuracy_score(y_test,p2)
accuracy3 = accuracy_score(y_test,p3)
accuracy4 = accuracy_score(y_test,p4)

print "Decision Tree : ", accuracy1 
print "K Nearest Neighbors : ", accuracy2 
print "Support Vector Machine : ", accuracy3
print "Logistic Regression : ", accuracy4