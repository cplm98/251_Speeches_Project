import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from FileHandler import FileHandler
from FeatureSelection import FeatureSelection
from sys import exit

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

# ## Try with normalized tag counts

# #l1 Test Accuracy:  0.8999999761581421
# l1_normalized_file = FileHandler("./l1_normalized_tags.csv", float)
# l1_normalized_file.read_csv()
# l1_normalized_tags = l1_normalized_file.get_data()

# # l1_normalized_data = data * l1_normalized_tags.reshape(1, l1_normalized_tags.size)

# # data_train, data_test, outcome_train, outcome_test = train_test_split(l1_normalized_data, winners_data, test_size=.3, random_state=42)
# # rows, cols = l1_normalized_data.shape

# # L2 Normalized
# # l2 Test Accuracy:  0.9076923131942749 -> reached 100 on test set by epoch 31, definitely some overfitting happening
# l2_normalized_file = FileHandler("./l2_normalized_tags.csv", float)
# l2_normalized_file.read_csv()
# l2_normalized_tags = l2_normalized_file.get_data()

# print(data.shape)
# print(l2_normalized_tags.reshape(1, l2_normalized_tags.size).shape)
# l2_normalized_data = data * l2_normalized_tags.reshape(1, l2_normalized_tags.size)
# print(l2_normalized_data.shape)

# data_train, data_test, outcome_train, outcome_test = train_test_split(l2_normalized_data, winners_data, test_size=.3, random_state=42)
# rows, cols = l2_normalized_data.shape

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1000, activation='relu'),
#     tf.keras.layers.Dense(1000/2, activation='relu'),
#     tf.keras.layers.Dense(2)
# ])

# model.compile(
#     optimizer = 'adam', # how model is updated based on loss function
#     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures accuracy of the model, and steering during training
#     metrics = ['accuracy'] # what it's using to measure success
# )

# # This works as a base, accuracy 92% with 20 epochs
# print(data_train[0][1])
# print("Training Model")
# print(data_train.shape)
# print(outcome_train.shape)
# print(type(data_train))
# print(type(outcome_train))
# # model.fit(data_train, outcome_train, epochs=100)
# # test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
# # print("Test Accuracy: ", test_acc)

# exit(0)







# use train_test_split twice to create 60 train, 20 validate, 20 test split
data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.2, random_state=42)
# data_train, data_val, outcome_train, outcome_val = train_test_split(data_train, outcome_train, test_size = .25, random_state=42)
rows, cols = data.shape

def test_connected_net(nums, data, outcomes, feature_selectors=None):
    res = {}
    for feature_selector in feature_selectors:    
        accuracies = []
        for num_feats in nums:
            attr_idx = feature_selector(num_feats)
            attributes = np.empty([rows, num_feats])
            for i, j in enumerate(attr_idx): # fill with important features columns
                attributes[:, i] = data[:, j]
            
            model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_feats, activation='relu'),
            tf.keras.layers.Dense(num_feats/2, activation='relu'),
            tf.keras.layers.Dense(2)
            ])

            model.compile(
                optimizer = 'adam', # how model is updated based on loss function
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures accuracy of the model, and steering during training
                metrics = ['accuracy'] # what it's using to measure success
            )  

            data_train, data_test, outcome_train, outcome_test = train_test_split(attributes, winners_data, test_size=.3, random_state=42)

            print("Training Model")
            model.fit(data_train, outcome_train, epochs=50)
            test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
            print("Test Accuracy: ", test_acc)
            accuracies.append(test_acc)
        res[str(feature_selector)] = accuracies
    print("\nRESULTS")
    print(nums)
    print(res)

    print("\n Accuracies by different number of features. Epochs 100")
    print(nums)
    print(accuracies)



# Base line, all features, uncomment to run

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

# This works as a base, accuracy 92% with 20 epochs
print("Training Model")
model.fit(data_train, outcome_train, epochs=100)
test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
print("Test Accuracy: ", test_acc)
pred=model.predict_classes(data_test)
print(tf.math.confusion_matrix(labels=outcome_test, predictions=pred).numpy())


feature_selector = FeatureSelection(data, winners_data)


### DRIVER CODE ####

# nums = [10, 100, 300, 500, 700, 800, 900, 1000]
# feature_selectors = [feature_selector.pearson_selector, feature_selector.chi_squared, feature_selector.recursive_feature_elim, feature_selector.rf_selector]
# test_connected_net(nums, data, winners_data, feature_selectors)

