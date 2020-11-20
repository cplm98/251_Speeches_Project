from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plt
from GeneralStats import GeneralStats

from sklearn.cluster import DBSCAN
# from sklearn import metrics
# from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, normalize


winners = FileHandler("./data/deceptionword.csv", str)
deception_wrd_matrix = FileHandler("./data/deceptiondocword.csv", int)
deception_wrd_matrix.read_csv()
data = deception_wrd_matrix.get_data()

# create general model for how deception words are used accross speaches

rows, cols = data.shape
col_sum = data.sum(axis = 0) # total use per word
row_sum = data.sum(axis = 1) # total deception per speach
avg_speach = col_sum / rows
# print("average speach: ", avg_speach)

# General Statistics
# per speach

# per_speach = GeneralStats(data)
# per_speach.get_statistics(1)

# # per word

# per_word = GeneralStats(data)
# per_word.get_statistics(0)

######################################
# x = range(cols)
# y = avg_speach
# plt.scatter(x, y,  s=2**2)
# plt.title("Average Deception Model")
# plt.show()

# cleaned_cols = {}
# for i, col in enumerate(col_sum):
#     if col == 0:
#         print
#         cleaned_cols[i] = data[:][i]

# print("Cleaned: ", len(cleaned_cols))
# print("Original: ", len(col_sum))
x_word = []
y_word = []
for row in range(rows):
    for col in range(cols):
        x_word.append(col)
        y_word.append(data[row][col]/col_sum[col])

# print("x: ", x_word)
# print("y: ", y_word)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_normalized = normalize(data_scaled)
# print(data_normalized)
db = DBSCAN(eps=.8, min_samples=5).fit(data_normalized)
print(db.labels_)
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)