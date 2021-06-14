import matplotlib.pyplot as plt
import csv
 
x = []
 
y_f1 = []
y_f1_nb = []
y_f1_maxent = []
y_f1_svm = []
 
var = [ ('all words', 'f1-all-words'), 
        ('all words+stopwords', 'f1-all-words-stopwords'), 
        ('bigrams', 'f1-bigrams'),
        ('bigrams+stopwords', 'f1-bigrams-stopwords'),        
      ]
 
my_x = []
 
for i,(key, val) in enumerate(var):    
    x.append(i+1)
    my_x.append(key)    
 
i = 0
with open('testset2.csv') as myfile:    
    reader = csv.DictReader(myfile, delimiter=',')    
    for line in reader:        
        if i == 0:
            for (key,val) in var:
                y_f1_nb.append(float(line[val]))
        elif i == 1:
            for (key,val) in var:
                y_f1_maxent.append(float(line[val]))
        elif i == 2:
            for (key,val) in var:
                y_f1_svm.append(float(line[val]))        
        i += 1
        
fig = plt.figure()
ax = plt.subplot(111) # row x col x position (here 1 x 1 x 1)
 
plt.xticks(x, my_x, rotation=75) # rotate x-axis labels to 75 degree
ax.plot(x, y_f1_nb, label='f1 - naive bayes', marker='o', linestyle='-', linewidth=1.5)
ax.plot(x, y_f1_maxent, label='f1 - maximum entropy', marker='o', linestyle='--', linewidth=1.5)
ax.plot(x, y_f1_svm, label='f1 - svm', marker='o', linestyle='-.', linewidth=1.5)
 
plt.xlim(0, len(var)+1)
plt.tight_layout() # showing xticks (as xticks have long names)
ax.grid()
 
plt.title('F1 measure plot', color='#000000', weight="bold", size="large")
plt.ylabel('F1 MEASURE')
plt.xlabel('FEATURES')
 
ax.legend(loc='lower right', fancybox=True, shadow=True)
 
plt.show()