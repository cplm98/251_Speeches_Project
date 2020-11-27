from FileHandler import FileHandler
from FeatureSelection import FeatureSelection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold 
from sys import exit

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

# save eval data as unseen data for testing at the end
data_cross, data_eval, outcome_cross, outcome_eval = train_test_split(data, winners_data, test_size=.2, random_state=42)


k = 8
kf = KFold(n_splits=k, shuffle=True, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000/2, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer = 'adam', # how model is updated based on loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures accuracy of the model, and steering during training
    metrics = ['accuracy'] # what it's using to measure success
)

acc_scores = []

for train_idx, test_idx in kf.split(data_cross):
    data_train, data_test = np.array([data_cross[i, :] for i in train_idx]), np.array([data_cross[i, :] for i in test_idx])
    outcome_train, outcome_test = np.array([outcome_cross[i] for i in train_idx]), np.array([outcome_cross[i] for i in test_idx])
    # print(data_train[0][1])
    # print(data_train.shape)
    # print(outcome_train.shape)
    # print(type(data_train))
    # print(type(outcome_train))
    model.fit(data_train, outcome_train, epochs=10)
    test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
    acc_scores.append(test_acc)

avg_acc_score = sum(acc_scores)/k

print('accuracy of each fold - {}'.format(acc_scores))
print('Avg accuracy : {}'.format(avg_acc_score))


test_loss, test_acc = model.evaluate(data_eval, outcome_eval, verbose=2)
print("Evaluation Accuracy: ", test_acc)
