import numpy as np
import matplotlib.pyplot as plt

Hernia= (60,62, 65, 57, 66, 57,97,56,20,58,71) 
Spondylolisthesis = (120,118, 115, 123, 114, 123,83,124,160,122,109)

ind = np.arange(len(Hernia))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(ind - width/2, Hernia, width,
                color='LightGreen', label='Hernia')
rects2 = ax.bar(ind + width/2, Spondylolisthesis, width,
                color='IndianRed', label='Spondylolithesis')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Disease')
ax.set_title('Classification of Orthopaedic Patients')
ax.set_xticks(ind)
ax.set_xticklabels(('ACTUAL','DT', 'KNN', 'RF', 'SVM', 'ADB','MLP','NB','QDA','LR','GP'))
ax.legend()