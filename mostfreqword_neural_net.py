import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from FileHandler import FileHandler
from FeatureSelection import FeatureSelection

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.3, random_state=42)
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
            model.fit(data_train, outcome_train, epochs=100)
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
# print("Training Model")
# model.fit(data_train, outcome_train, epochs=100)
# test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
# print("Test Accuracy: ", test_acc)


feature_selector = FeatureSelection(data, winners_data)


### DRIVER CODE ####

nums = [10, 50, 100, 200, 300, 500, 700, 800]
feature_selectors = [feature_selector.pearson_selector, feature_selector.chi_squared, feature_selector.rf_selector]
test_connected_net(nums, data, winners_data, feature_selectors)

