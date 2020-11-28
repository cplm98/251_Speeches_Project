from FileHandler import FileHandler
from FeatureSelection import FeatureSelection
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from FileHandler import FileHandler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sys import exit

# try with different kernels

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.2, random_state=42)

# kernels = ['poly', 'rbf']
# coeffs = [.5, .75, 1, 3, 10, 15]

# kernels=['poly']
# coeffs = [10]

# for kernel in kernels:
#     print("\nKernel: ", kernel)
#     for coeff in coeffs:
#         model = SVC(C=coeff, kernel = kernel)
#         model.fit(data_train, outcome_train)
#         print("\nTrain Accuracy: ", model.score(data_train, outcome_train))
#         print("Test Accuracy: ", model.score(data_test, outcome_test))

# exit(0)

def test_svm(data, outcomes, nums, feature_selectors, k):
    rows, cols = data.shape
    res = {}
    for feature_selector in feature_selectors:
        accuracies = []
        for num_feats in nums:
            attr_idx = feature_selector(num_feats)
            attributes = np.empty([rows, num_feats])
            for i, j in enumerate(attr_idx):
                attributes[:, i] = data[:, j]

            data_train, data_test, outcome_train, outcome_test = train_test_split(attributes, winners_data, test_size=.2, random_state=42)

            svc_model = SVC(C=10, kernel = 'poly')
            svc_model.fit(data_train, outcome_train)
            predictions = svc_model.predict(data_test)
            print("\nTrain Accuracy: ", svc_model.score(data_train, outcome_train))
            print("Test Accuracy: ", svc_model.score(data_test, outcome_test))
            accuracies.append(svc_model.score(data_test, outcome_test))
        res[str(feature_selector)] = accuracies
    print(nums)
    print(res)


feature_selector = FeatureSelection(data, winners_data)
nums = [10, 100, 300, 500, 700, 800, 900, 1000]
feature_selectors = [feature_selector.pearson_selector, feature_selector.chi_squared, feature_selector.recursive_feature_elim, feature_selector.rf_selector]
test_svm(data, winners_data, nums, feature_selectors, 5)    
