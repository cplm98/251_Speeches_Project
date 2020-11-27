from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plt
from GeneralStats import GeneralStats

# from sklearn.cluster import DBSCAN
# # from sklearn import metrics
# # from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler, normalize


winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

deception_wrd_matrix = FileHandler("./data/deceptiondocword.csv", int)
deception_wrd_matrix.read_csv()
data = deception_wrd_matrix.get_data()

# create general model for how deception words are used accross speaches

rows, cols = data.shape
col_sum = data.sum(axis = 0) # total use per word
row_sum = data.sum(axis = 1) # total deception per speach
avg_speach = col_sum / rows
# print("average speach: ", avg_speach)

# print("Deception Column Stats")
# col_stats = GeneralStats(col_sum)
# col_stats.get_statistics(0)

# print("Deception Row Stats")
# row_stats = GeneralStats(row_sum)
# row_stats.get_statistics(0)

winning_speeches = []
losing_speeches = []
for row in range(rows):
    if winners_data[row] == 1:
        winning_speeches.append(data[row, :])
    else:
        losing_speeches.append(data[row, :])

winning_speeches = np.array(winning_speeches)
losing_speeches = np.array(losing_speeches)

winning_cols = winning_speeches.sum(axis = 0)
wcols_stats = GeneralStats(winning_cols)
print("Winning Cols: ")
wcols_stats.get_statistics(0)

print("\nLength: ", len(winning_speeches[:]))
winning_rows = winning_speeches.sum(axis = 1) #/ len(winning_speeches[:])
wrows_stats = GeneralStats(winning_rows)
print("\nWinning Rows:")
wrows_stats.get_statistics(0)

losing_cols = losing_speeches.sum(axis = 0)
lcols_stats = GeneralStats(losing_cols)
print("\nLosing Cols: ")
lcols_stats.get_statistics(0)

losing_rows = losing_speeches.sum(axis = 1) #/ len(losing_speeches[:])
lrows_stats = GeneralStats(losing_rows)
print("\nLosing Rows:")
lrows_stats.get_statistics(0)

# print("Winners: ", winning_speeches.shape)
# print("Losers: ", losing_speeches.shape)

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
# x_word = []
# y_word = []
# for row in range(rows):
#     for col in range(cols):
#         x_word.append(col)
#         y_word.append(data[row][col]/col_sum[col])

# print("x: ", x_word)
# print("y: ", y_word)

# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)
# data_normalized = normalize(data_scaled)
# print(data_normalized)
# db = DBSCAN(eps=.8, min_samples=5).fit(data_normalized)
# print(db.labels_)
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)