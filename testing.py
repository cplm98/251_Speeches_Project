from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plt
from GeneralStats import GeneralStats

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

deception_wrd_matrix = FileHandler("./data/deceptiondocword.csv", int)
deception_wrd_matrix.read_csv()
data = deception_wrd_matrix.get_data()

rows, cols = data.shape
col_sum = data.sum(axis = 0) # total use per word
row_sum = data.sum(axis = 1) # total deception per speach

print("Col sum: ", col_sum)

# number of deceptions vs outcome

raw_win_sum = 0
raw_lose_sum = 0
win_deceipts = 0
lose_deceipts = 0
for i, outcome in enumerate(winners_data):
    if outcome == 0:
        raw_lose_sum += 1
        lose_deceipts += row_sum[i]
    else:
        raw_win_sum += 1
        win_deceipts += row_sum[i]

print("Total Winner Deceipts: ", win_deceipts)
print("Total Loser Deceipts: ", lose_deceipts)

print("Deceipts per Winner: ", (win_deceipts/raw_win_sum))
print("Deceipts per Loser: ", (lose_deceipts/raw_lose_sum))
print("Win/Loss Deceipt Ratio", ((win_deceipts/raw_win_sum)/(lose_deceipts/raw_lose_sum)), "\n")

# The graphs work and are interesting
# plt.scatter(winners_data, row_sum, 2**2)
# plt.autoscale(enable=True, axis='x', tight=False)
# plt.xticks(range(0,2))
# plt.show()

# normalized number of deceptions versus outcomes
data_normalized = data / col_sum # normalize by count per word
normalized_row_sum = data_normalized.sum(axis=1)
# plt.scatter(winners_data, normalized_row_sum, 2**2)
# plt.xticks(range(0,2))
# plt.show()

normalized_win_sum = 0
normalized_lose_sum = 0
normalized_lose_deceipts = 0
normalized_win_deceipts = 0
for i, outcome in enumerate(winners_data):
    if outcome == 0:
        normalized_lose_sum += 1
        normalized_lose_deceipts += normalized_row_sum[i]
    else:
        normalized_win_sum += 1
        normalized_win_deceipts += normalized_row_sum[i]

print("Total Winner Deceipts: ", normalized_win_deceipts)
print("Total Loser Deceipts: ", normalized_lose_deceipts)

print("Deceipts per Winner: ", (normalized_win_deceipts/normalized_win_sum))
print("Deceipts per Loser: ", (normalized_lose_deceipts/normalized_lose_sum))
print("Normalized Win/Loss Ratio: ", ((normalized_win_deceipts/normalized_win_sum)/(normalized_lose_deceipts/normalized_lose_sum)))