import pandas as t
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
from matplotlib import rcParams
#import numpy as np

#data = t.read_csv('column_2C_weka.csv')
#feature_cols = ["pelvic_incidence","pelvic_tilt numeric","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis"]
#X = data[feature_cols]
#label_cols = ["class"]
#y = data[label_cols]

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.65)

traindata = t.read_csv('trainset2.csv')
testdata= t.read_csv('testset2.csv')

feature_cols = ["pelvic_incidence","pelvic_tilt numeric","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis"]

label_cols = ["class"]



y_train = traindata[label_cols]
y_test = testdata[label_cols]
X_train = traindata[feature_cols]
X_test = testdata[feature_cols]


dtc =DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train,y_train)

predictions = dtc.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)

print(conf_matrix)
lst = []
lst2= []
W_test= []
z_test = []
#conf_matrix = confusion_matrix(y_test, predictions)

#print(conf_matrix)
analysis1 = accuracy_score(y_test,predictions)
print(analysis1)



#converting string to float
le = LabelEncoder()
le.fit(y_test)
yorg = le.transform(y_test)


le = LabelEncoder()
le.fit(predictions)
y = le.transform(predictions)


m=[]
z=[]
i=0
while i<len(y):
    z.append(y[i])
    m.append(yorg[i])
    i=i+1
    

x=[]
n=0
while n<len(z):
    x.append(n)
    n=n+1
    
print(m)
print(z)

plt.plot(x,y,'-',lw=2, color='#eeefff')

plt.plot(x,m,'-',lw=2,color='y')
plt.xlabel('testset')
plt.ylabel('Hernia/Spondylolisthesis')    
plt.show()    

#plt.plot(x, z, color='g')
#plt.show()


#print(predictions)


##print(length)
#s = 'Abnormal'
#
#
##s1 = 'Normal'
#count = 0
##count2=0
#
#i = -1
#while i <= length-1 :
#    if s == predictions[i]:
#      lst.append(predictions[i])
#      lst2.append(X_test.iloc[[i]])
#      count=count + 1
#      
#      #print(i)
#      i=i+1
#      
#      
#    else: 
#         i=i+1
#         
#    
##i = 0
##while i < length :
##    if s1 == predictions[i]:
##     count2=count2 + 1
##     i=i+1
##    else: 
##         i=i+1
#j=-1
#while j<=len(lst2)-1:
#    W_test= t.DataFrame(data=W_test.append(lst2[j]))
#    j=j+1
#    
#     
#z_test= t.DataFrame(data=lst)
##print(W_test.shape)    
#print(count)
#
##print(len(lst))
##print(lst)
##print(count2)
#
#trainset= t.read_csv('tainset2.csv')       
#
#         
#z_train=trainset[label_cols]
#
#W_train=trainset[feature_cols]
#
#
#dtc.fit(W_train,z_train)
#
#predictions2 = dtc.predict(W_test)
#
#analysis2=accuracy_score(z_test,predictions2)
#print(analysis2)
#
##print(predictions2)
