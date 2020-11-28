from FileHandler import FileHandler
from GeneralStats import GeneralStats
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sys import exit

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

# words = FileHandler("./data/mostfreq1000word.csv", str)
# words.read_csv()
# words_list = words.get_data()

words_raw = np.genfromtxt("./data/mostfreq1000word.csv", dtype='str', encoding='latin-1')

data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.2, random_state=42)
rows, cols = data.shape

num_trees = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
depth = [1, 10, 25, 50, 100]
# for num in num_trees:
#     rf = RandomForestClassifier(n_estimators = num, max_features = num, random_state = 42)
#     rf.fit(data_train, outcome_train)
#     pred = rf.predict(data_test)
#     print("Num Trees:", num)
#     print("ACCURACY OF THE MODEL ALL FEATURES: ", metrics.accuracy_score(outcome_test, pred)) 

rf = RandomForestClassifier(n_estimators = 100, max_features = 600, random_state = 42)
rf.fit(data_train, outcome_train)
pred = rf.predict(data_test)
conf_matx = confusion_matrix(outcome_test, pred)
print(conf_matx)
cr = classification_report(outcome_test, pred)
print(cr) 
importance_matx = rf.feature_importances_
# print("\nImportance", importance_matx)
print(importance_matx.shape)
top_all = np.argsort(importance_matx)[-600:]
vals_all = np.array([importance_matx[i] for i in top_all])
top_idx = np.argsort(importance_matx)[-25:]
top_vals = np.array([importance_matx[i] for i in top_idx])
print(np.argsort(importance_matx)[-25:])

print(top_vals)

# for i in top_idx:
#     # print(sum(data[:, i]))
#     # print(words_raw[i])

types_of_speech = {}
for i in top_all:
    word, tag = words_raw[i].split("_")
    if tag in types_of_speech:
        types_of_speech[tag] += importance_matx[i]
    else:
        types_of_speech[tag] = importance_matx[i]

print(types_of_speech.keys())
print(types_of_speech.values())

useage_stats = GeneralStats(vals_all)
useage_stats.get_statistics(0)

