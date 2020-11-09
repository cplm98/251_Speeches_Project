from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plts

winners = FileHandler("./data/deceptionword.csv", str)
most_freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
most_freq_wrd_matrix.read_csv()
data = most_freq_wrd_matrix.get_data()

col_sum = data.sum(axis = 0)
top_five = np.argpartition(col_sum, -10)[-10:]
print(col_sum.shape)
print("Column sum: ", top_five)
print(col_sum[top_five])


