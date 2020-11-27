from FileHandler import FileHandler
from FeatureSelection import FeatureSelection
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from FileHandler import FileHandler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sys import exit

# https://www.datatechnotes.com/2020/06/classification-example-with-svc-in-python.html

winners = FileHandler("./data/winners.csv", int)
winners.read_csv()
winners_data = winners.get_data()

freq_wrd_matrix = FileHandler("./data/mostfreq1000docword.csv", float)
freq_wrd_matrix.read_csv()
data = freq_wrd_matrix.get_data()

# data_train, data_test, outcome_train, outcome_test = train_test_split(data, winners_data, test_size=.3, random_state=42)
rows, cols = data.shape

#Chi2 Feature Selection
num_feats = 800
feature_selector = FeatureSelection(data, winners_data)
chi2_attr_idx = feature_selector.chi_squared(num_feats)
chi2_data = np.empty([rows, num_feats])
for i, j in enumerate(chi2_attr_idx): # fill with important features columns
    chi2_data[:, i] = data[:, j]

data_train, data_test, outcome_train, outcome_test = train_test_split(chi2_data, winners_data, test_size=.3, random_state=42)

print(chi2_data.shape)
print(data_train.shape)

svc_model = SVC()
svc_model.fit(data_train, outcome_train)
score = svc_model.score(data_train, outcome_train)
print("Score: ", score)

outcome_predction = svc_model.predict(data_test)
conf_matx = confusion_matrix(outcome_test, outcome_predction)
print(conf_matx)
cr = classification_report(outcome_test, outcome_predction)
print(cr)
