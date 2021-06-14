import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches

traindata=pd.read_csv("trainset2.csv")
testdata=pd.read_csv("testset2.csv")

features = ["pelvic_incidence","pelvic_tilt numeric","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis"]
labels = ["class"]

y_train = traindata[labels]
y_test = testdata[labels]
X_train = traindata[features]
X_test = testdata[features]

le = LabelEncoder()
le.fit(y_test)
actual = le.transform(y_test)

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train,y_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train.values.ravel())

rfc= RandomForestClassifier()
rfc.fit(X_train,y_train.values.ravel())

svm =  SVC(kernel="linear", C=0.005)
svm.fit(X_train,y_train)

abc = AdaBoostClassifier()
abc.fit(X_train,y_train)

mlp = MLPClassifier()
mlp.fit(X_train,y_train)

nbc = GaussianNB()
nbc.fit(X_train,y_train)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train,y_train)

lrc = LogisticRegression()
lrc.fit(X_train,y_train)

gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
gpc.fit(X_train,y_train.values.ravel()) 

p1 = dtc.predict(X_test)
p2 = knn.predict(X_test)
p3 = rfc.predict(X_test)
p4 = svm.predict(X_test)
p5 = abc.predict(X_test)
p6 = mlp.predict(X_test)
p7 = nbc.predict(X_test)
p8 = qda.predict(X_test)
p9 = lrc.predict(X_test)
p10 = gpc.predict(X_test)

le = LabelEncoder()
le.fit(p1)
a = le.transform(p1)
le.fit(p2)
b = le.transform(p2)
le.fit(p3)
c = le.transform(p3)
le.fit(p4)
d = le.transform(p4)
le.fit(p5)
e = le.transform(p5)
le.fit(p6)
f = le.transform(p6)
le.fit(p7)
g = le.transform(p7)
le.fit(p8)
h = le.transform(p8)
le.fit(p9)
i = le.transform(p9)
le.fit(p10)
j = le.transform(p10)

count=[]
n=0
while n<len(y_test):
    count.append(n)
    n=n+1
    
fig = plt.figure(figsize=(10,5))
#fig.suptitle('Comparison between prediction and actual result', fontsize='13')
#    
#plt.plot(count,f,color='firebrick',alpha=0.9)
#
##plt.plot(count,b,color='g')
##plt.plot(count,c,color='g')
##plt.plot(count,d,color='g')
##plt.plot(count,e,color='y')
##plt.plot(count,f,color='w')
##plt.plot(count,g,color='k')
##plt.plot(count,h,color='m')
##plt.plot(count,i,color='c')
##plt.plot(count,j,color='r')
#plt.plot(count,actual,color='black',alpha=0.9, label='a')
#plt.xlabel('Number of Patients',fontsize='10.5')
#plt.ylabel('Hernia=0 , Spondylolisthesis=1',fontsize='10.5')
#black_patch = mpatches.Patch(color='black', label='Actual')
#red_patch= mpatches.Patch(color='firebrick', label='MLP')
#plt.legend(handles=[red_patch,black_patch])
#plt.show()
#)

plt.plot(j,color='firebrick',alpha=0.9, label="GP")
plt.plot(actual,color='black',alpha=0.9, label="Actual")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Number of Patients',fontsize='11.5')
plt.ylabel('Hernia=0 , Spondylolisthesis=1',fontsize='11.5')
plt.show()
fig.savefig('GP.jpg')
            
            
accuracy1 = accuracy_score(y_test,p1)
accuracy2 = accuracy_score(y_test,p2)
accuracy3 = accuracy_score(y_test,p3)
accuracy4 = accuracy_score(y_test,p4)
accuracy5 = accuracy_score(y_test,p5)
accuracy6 = accuracy_score(y_test,p6)
accuracy7 = accuracy_score(y_test,p7)
accuracy8 = accuracy_score(y_test,p8)
accuracy9 = accuracy_score(y_test,p9)
accuracy10 = accuracy_score(y_test,p10)

matrix1 = confusion_matrix(y_test, p1)
matrix2 = confusion_matrix(y_test, p2)
matrix3 = confusion_matrix(y_test, p3)
matrix4 = confusion_matrix(y_test, p4)
matrix5 = confusion_matrix(y_test, p5)
matrix6 = confusion_matrix(y_test, p6)
matrix7 = confusion_matrix(y_test, p7)
matrix8 = confusion_matrix(y_test, p8)
matrix9 = confusion_matrix(y_test, p9)
matrix10 = confusion_matrix(y_test, p10)

print("Decision Tree : ", accuracy1) 
print("K Nearest Neighbors : ", accuracy2) 
print("Random Forest : ", accuracy3) 
print("Support Vector Machine : ", accuracy4) 
print("Adaptive Boosting : ", accuracy5) 
print("Multilayer Perception : ", accuracy6) 
print("Naive Bayes : ", accuracy7) 
print("Quadratic Discriminant : ", accuracy8)
print("Logistic Regression : ", accuracy9) 
print("Gaussian Process : ", accuracy10) 

print("Decision Tree : ", matrix1) 
print("K Nearest Neighbors : ", matrix2) 
print("Random Forest : ", matrix3) 
print("Support Vector Machine : ", matrix4) 
print("Adaptive Boosting : ", matrix5) 
print("Multilayer Perception : ", matrix6) 
print("Naive Bayes : ", matrix7) 
print("Quadratic Discriminant : ", matrix8)
print("Logistic Regression : ", matrix9) 
print("Gaussian Process : ", matrix10) 