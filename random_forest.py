from FileHandler import FileHandler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.3, random_state=42)
rows, cols = data.shape

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 10000, random_state = 42)
# Train the model on training data
rf.fit(data_train, outcome_train)

predictions = rf.predict(data_test) # comes out as a confidence score I belive
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


