from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plts
from GeneralStats import GeneralStats

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

most_freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
most_freq_wrd_matrix.read_csv()
data = most_freq_wrd_matrix.get_data()
rows, cols = data.shape

col_sum = data.sum(axis = 0)
row_sum = data.sum(axis = 1)
# print(row_sum)
# top_five = np.argpartition(col_sum, -10)[-10:]
# print(col_sum.shape)
# print("Column sum: ", top_five)
# print(col_sum[top_five])
# print("Column wise std: ", np.std(col_sum))

# freq_stats = GeneralStats(col_sum)
# # freq_stats.get_statistics(0)
# print("Column Stats")
# freq_stats.get_statistics(0)
# print("Row Stats")
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
winning_rows = winning_speeches.sum(axis = 1) # / len(winning_speeches[:])
wrows_stats = GeneralStats(winning_rows)
print("\nWinning Rows:")
wrows_stats.get_statistics(0)

losing_cols = losing_speeches.sum(axis = 0)
lcols_stats = GeneralStats(losing_cols)
print("\nLosing Cols: ")
lcols_stats.get_statistics(0)

losing_rows = losing_speeches.sum(axis = 1) # / len(losing_speeches[:])
lrows_stats = GeneralStats(losing_rows)
print("\nLosing Rows:")
lrows_stats.get_statistics(0)




