# accuracies = []
# for num_feats in nums:
#     # attr_idx = feature_selector.pearson_selector(num_feats) # get the sorted index of important features
#     # attr_idx = feature_selector.chi_squared(num_feats)
#     # attr_idx = feature_selector.recursive_feature_elim(num_feats)
#     attr_idx = feature_selector.rf_selector(num_feats)
#     # pearson_attributes = np.empty([rows, num_feats])
#     # chi2_attributes = np.empty([rows, num_feats])
#     # rfe_attributes = np.empty([rows, num_feats])
#     rf_attributes = np.empty([rows, num_feats])
#     for i, j in enumerate(attr_idx): # fill with important features columns
#         # pearson_attributes[:, i] = data[:, j]
#         # chi2_attributes[:, i] = data[:, j]
#         # rfe_attributes[:, i] = data[:, j]
#         rf_attributes[:, i] = data[:, j]
    
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense(num_feats, activation='relu'),
#     tf.keras.layers.Dense(num_feats/2, activation='relu'),
#     tf.keras.layers.Dense(2)
#     ])

#     model.compile(
#         optimizer = 'adam', # how model is updated based on loss function
#         loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures accuracy of the model, and steering during training
#         metrics = ['accuracy'] # what it's using to measure success
#     )  

#     # Pearson
#     # data_train, data_test, outcome_train, outcome_test = train_test_split(pearson_attributes, winners_data, test_size=.3, random_state=42)
#     # data_train, data_test, outcome_train, outcome_test = train_test_split(chi2_attributes, winners_data, test_size=.3, random_state=42)
#     # data_train, data_test, outcome_train, outcome_test = train_test_split(rfe_attributes, winners_data, test_size=.3, random_state=42)
#     data_train, data_test, outcome_train, outcome_test = train_test_split(rf_attributes, winners_data, test_size=.3, random_state=42)


#     print("Training Model")
#     model.fit(data_train, outcome_train, epochs=100)
#     test_loss, test_acc = model.evaluate(data_test, outcome_test, verbose=2)
#     print("Test Accuracy: ", test_acc)
#     accuracies.append(test_acc)

# print("\n Accuracies by different number of features. Epochs 100")
# print(nums)
# print(accuracies)

# print("Pearson: ", feature_selector.pearson_selector(10))
# print("Chi2: ", feature_selector.chi_squared(10))
# print("RFE: ", feature_selector.recursive_feature_elim(10))
# print("Lasso: ", feature_selector.lasso_selector(10))
# print("Random Forest: ", feature_selector.rf_selector(10))