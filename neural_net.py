import tensorflow as tf
# from tensorflow.keras import layers, Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from FileHandler import FileHandler

print("Done Importing")

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

deception_wrd_matrix = FileHandler("./data/deceptiondocword.csv", int)
deception_wrd_matrix.read_csv()
data = deception_wrd_matrix.get_data()

data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.3, random_state=42)
rows, cols = data.shape

print("Split Complete")
print(len(data_test))
print(len(outcome_test), "\n")

print("Building Model")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(76, activation='relu'),
    tf.keras.layers.Dense(76/2, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer = 'adam', # how model is updated based on loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures accuracy of the model, and steering during training
    metrics = ['accuracy'] # what it's using to measure success
)

# This works as a base, accuracy 77% with 10 epochs
print("Training Model")
model.fit(data_train, outcome_train, epochs=10)
test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
print("Test Accuracy: ", test_acc)

# do some kind of feature selection

#https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2
# pearson correlation


feature_selection_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

feature_selection_model.compile(
    optimizer = 'adam', # how model is updated based on loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures accuracy of the model, and steering during training
    metrics = ['accuracy'] # what it's using to measure success
)

feature_importance = {}
for col in range(cols):
    data_set = data[:, col]
    data_train, data_test, outcome_train, outcome_test = train_test_split(data_set, winners_data, test_size=.3, random_state=42)
    feature_selection_model.fit(data_train, outcome_train, epochs=5)
    test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
    feature_importance[col] = test_acc

prev_test_acc = 0
num_features = 0
sorted_features = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1])}
sorted_features_index = sorted_features.keys
print("sorted_features_index: ", sorted_features_index)
# while True:


