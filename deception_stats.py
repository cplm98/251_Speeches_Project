from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plt
from GeneralStats import GeneralStats

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

per_speach = GeneralStats(data)
per_speach.get_statistics(1)

# per word

per_word = GeneralStats(data)
per_word.get_statistics(0)

######################################
# x = range(cols)
# y = avg_speach
# plt.scatter(x, y,  s=2**2)
# plt.title("Average Deception Model")
# plt.show()