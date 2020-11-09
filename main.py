from FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

winners = FileHandler("./data/deceptionword.csv", str)
most_freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
# winners.read_csv()
most_freq_wrd_matrix.read_csv()
data = most_freq_wrd_matrix.get_data()

# scatter
# x = range(1, 1001)
# y = data[0][:]

five_max = np.argpartition(data[0], -5)[-5:]
# print(five_max)
# print(data[0][five_max])

top_five = np.empty([431, 5], dtype=int)
print(data.shape)
y = np.empty([431, 5])
rows, cols = data.shape
for row in range(rows):
    top_five[row] = np.argpartition(data[row], -5)[-5:]
    # print(top_five[row])
    y[row] = data[0][top_five[row]]

# print(top_five)
# y's
# print(y)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_ylabel("Rate of Use")
ax.set_xlabel("mostfreq Word Index")
cmap = get_cmap(431)
for i in range(431):
    ax.scatter(top_five[i], y[i], color=cmap(i))

# for lst in top_five:
#     if any(y > 4 for y in lst):
#         print(lst)

plt.title("Top 5 Most Frequently used Words Across Speeches")
plt.show()