from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plts
from GeneralStats import GeneralStats
from sys import exit
import re

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

most_freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
most_freq_wrd_matrix.read_csv()
data = most_freq_wrd_matrix.get_data()
rows, cols = data.shape

speeches_csv = FileHandler("./data/speeches.csv", str)
speeches_csv.read_csv()
speeches_names = speeches_csv.get_data()

speech_counts = {}
for i, speech in enumerate(speeches_names):
    tmp = speech[4:]
    # print(tmp)
    name = ""
    for char in tmp:
        if char.isalpha():
            name += char
        else:
            break
    if name in speech_counts:
        speech_counts[name].append(i)
    else:
        speech_counts[name] = [i]
    
# for candidate in speech_counts:
#     string = candidate + '_speech_indeces.csv'
#     print(string)
#     np.savetxt(string, speech_counts[candidate], delimiter=',', fmt='%s')

for candidate in speech_counts:
    arr = speech_counts[candidate]
    candidate_records = np.empty((len(arr), 1000))
    for i, j in enumerate(arr):
                candidate_records[i, :] = data[j, :]
    col_sum = candidate_records.sum(axis = 0) / len(arr)
    row_sum = candidate_records.sum(axis = 1)
    ccol_stats = GeneralStats(col_sum)
    print("\n" + candidate + " Column Stats")
    ccol_stats.get_statistics(0)
    crow_stats = GeneralStats(row_sum)
    print(candidate + " Row Stats")
    crow_stats.get_statistics(0)
    

exit(0)

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




