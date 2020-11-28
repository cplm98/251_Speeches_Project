from FileHandler import FileHandler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sys import exit

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.2, random_state=42)
rows, cols = data.shape

rf = RandomForestClassifier(n_estimators = 500, random_state = 42)
rf.fit(data_train, outcome_train)
pred = rf.predict(data_test)
print("ACCURACY OF THE MODEL ALL FEATURES: ", metrics.accuracy_score(outcome_test, pred)) 

sel = SelectFromModel(RandomForestClassifier(n_estimators = 1000))
sel.fit(data_train, outcome_train)
# sel.get_support()
# print(sel.get_support())
selected_feat = []
for i, x in enumerate(sel.get_support()):
    if x == True:
        selected_feat.append(i)
# print(selected_feat)

feat_sel_data = np.empty([rows,len(selected_feat)])
print(feat_sel_data.shape)
for i, j in enumerate(selected_feat): # fill with important features columns
    feat_sel_data[:, i] = data[:, j]

data_train, data_test, outcome_train, outcome_test = train_test_split(feat_sel_data, winners_data, test_size=.2, random_state=42)


rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(data_train, outcome_train)
pred = rf.predict(data_test)
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(outcome_test, pred)) 


exit(0)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 10000, random_state = 42)
# Train the model on training data
rf.fit(data_train, outcome_train)

predictions = rf.predict(data_test)
print(predictions)
print("predictions shape", predictions.shape)
print("outcomes shape: ", outcome_test.shape)
print("Correct Guesses: ", -sum(abs(predictions - outcome_test))%130, "/", len(outcome_test))
print("Accuracy: ", (-sum(abs(predictions - outcome_test))%len(outcome_test))/len(outcome_test))
# acc = (1 - sum(abs(predictions - outcome_test))) / len(outcome_test)
# print("Accuracy: ", acc)
# errors = abs(predictions - outcome_test)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / outcome_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')


