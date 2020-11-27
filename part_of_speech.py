from FileHandler import FileHandler
from FeatureSelection import FeatureSelection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sys import exit

# most_freq_word = FileHandler("./data/mostfreq1000word.csv", str)
# most_freq_word.read_csv()
# words_raw = most_freq_word.get_data()

words_raw = np.genfromtxt("./data/mostfreq1000word.csv", dtype='str', encoding='latin-1')

word_pairs = {}
words = []
tags = []
for pair in words_raw:
    word, tag = pair.split("_")
    if word:
        words.append(word)
        tags.append(tag)
        word_pairs[word] = tag
    else: # skip anything not encoded properly
        print(here)
        continue

tags_dict = {}
for tag in tags:
    if tag in tags_dict:
        tags_dict[tag] += 1
    else:
        tags_dict[tag] = 1

tags_all = np.array(list(tags_dict.keys()))
tag_counts_all = np.array(list(tags_dict.values()))

print(tag_counts_all)
print(sum(tag_counts_all))
print(max(tag_counts_all)/min(tag_counts_all))
print(np.ptp(tag_counts_all))

# normalized_tag_counts = normalize([tag_counts_all], norm='l1') # used for l1 and l2

# https://www.youtube.com/watch?v=FDCfw-YqWTE
tag_counts_mean = np.mean(tag_counts_all)
print(tag_counts_mean)
mu = (1/tag_counts_mean) * tag_counts_all
mean_adjusted_tag_counts = tag_counts_all - mu
sigma = (1/tag_counts_mean) * np.square(mean_adjusted_tag_counts)
normalized_tag_counts = mean_adjusted_tag_counts / sigma # this maintains same difference between min and max, but scales it down
# print(normalized_tag_counts)
# print(sum(normalized_tag_counts))
# print(max(normalized_tag_counts)/min(normalized_tag_counts))
# print(np.ptp(normalized_tag_counts))

# Standardization
# tag_counts_std = np.std(tag_counts_all)
# stadardized_tag_counts = abs((tag_counts_all - tag_counts_mean) / tag_counts_std)
# print(stadardized_tag_counts)
# print(sum(stadardized_tag_counts))
# print(max(stadardized_tag_counts)/min(stadardized_tag_counts))

#min max this gives divide by 0 errors which makes sense
# tag_counts_min = min(tag_counts_all)
# tag_counts_max = max(tag_counts_all)
# min_max_tags = (tag_counts_all - tag_counts_min) / (tag_counts_max - tag_counts_min)
# print(min_max_tags)
# print(sum(min_max_tags))
# print(max(min_max_tags)/min_max_tags)

exit(0)

normalized_for_csv = []
for tag in tags:
    i = tags_all.index(tag)
    normalized_for_csv.append(normalized_tag_counts[0][i])

print(normalized_for_csv)
print(len(normalized_for_csv))

# np.savetxt('l1_normalized_tags.csv', normalized_for_csv, delimiter=',', fmt='%s')

# plt.bar(tags_for_plot, tag_counts, align='center', alpha=0.5)
# plt.show()

fifty_percentile = np.percentile(tag_counts_all, 50)
filtered_tag_counts = []
filtered_tags = []
for i, tag in enumerate(tag_counts):
    if tag > fifty_percentile:
        filtered_tag_counts.append(tag)
        filtered_tags.append(tags_for_plot[i])

# print(filtered_tags)
# print(filtered_tag_counts)
# plt.bar(filtered_tags, filtered_tag_counts, align='center', alpha=0.5)
# plt.show()   

