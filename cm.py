import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Original confusion matrix data
cm = np.array([[  19633    1174    2456   18819    5336      89       5       0     914        0       0      19    1169     237       0       0]
 [   1153  445911    5365  910295    6107     920     246       0    2996        0       0       0    4533     266       0       0]
 [    820   23259  140752  730778   19214     543     443       0    2504        0       0       0    1205     455       0       0]
 [     77   77558   12565 4800414   29820    1212     301       1    3396        0       0      56    7618     158       0       0]
 [    101    4375    1956  197574  608854      33      44      12    1889        0       0      17    3038      53       0       0]
 [      2     977     225   29407     372   15753       1       0     170        0       0       0      84       0       0       0]
 [      0       0     322    7023      10       0   12870       0       0        0       0       0      25       5       0       0]
 [      0     744     372   55755     752      33       2    4289      44        0       0      15     156       0       0       0]
 [    278    6637     538  119898   13056     505       0       0   29533        0       0     153    3008       0       0       0]
 [      0     257      11    5914   10313       5       0       0     255       46       0       0      71     350       0       0]
 [      2     158      26    8315    1442       8       0       0      16        0       0       0      30       0       0       0]
 [      0    5723       8   48392     392       3       0      81     736        0       0   11771      26       0       0       0]
 [     34     940    1461   71014    6807       0     120       0     106        0       0       1  124152      18       0       0]
 [     92     390   15897   52427    4765       0     221       0      10        0       0       0    2544   45830       0       0]
 [      2     123    1530   16336     910       0     232       0       1        0       0       0    1796       5       0       0]
 [      0       3      89   38051     415       0       5       0       2        0       0       0     624       0       0       1]])

# Convert to percentages
row_sums = cm.sum(axis=1, keepdims=True)
cm_percentages = (cm / row_sums) * 100

# Create and save percentage-based confusion matrix plot
plt.figure(figsize=(12,10))
sns.heatmap(cm_percentages, annot=True, fmt='.1f', cmap='Blues')
plt.title('Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save plot
plt.savefig('confusion_matrix_percentages.png', bbox_inches='tight', dpi=300)
plt.close()

# Print percentage matrix
print("\nConfusion Matrix (Percentages):")
print(np.array2string(cm_percentages, precision=1, suppress_small=True))