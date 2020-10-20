import math
import numpy as np
import pickle
import matplotlib.pyplot as plt

VARIABILITY_THRESHOLD = 1450

def distance_L2(nss, trace):
    return np.linalg.norm(np.array(nss) - np.array(trace))

def variability(nss):
    return np.std(nss)

data = []

with open('silkroad_NSS_vs_trace.pkl', 'rb') as data_file:
    data = pickle.load(data_file)
    
all_users = []
all_variable_users = []
confusion_matrix = []


for i in data:
    user_name, nss, _ = i
    all_users.append(user_name)
    if variability(nss) < VARIABILITY_THRESHOLD:
        continue
    all_variable_users.append(user_name)

    curr_row = []
    for j in data:
        _, _, trace = j
        curr_row.append(int(distance_L2(nss, trace)))
    confusion_matrix.append(curr_row)

fig = plt.figure(figsize=(25,4))
ax = fig.add_subplot(1,1,1)
table = ax.table(cellText=confusion_matrix, colLabels=all_users, rowLabels=all_variable_users,loc='center')
table.auto_set_font_size(False)
table.set_fontsize(6.5)
table.auto_set_column_width([i for i in range(len(all_users))])
ax.axis('off')
ax.set_title('Silkroad confusion matrix: variable NSS vs. trace')

plt.show()



