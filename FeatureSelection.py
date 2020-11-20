from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


class FeatureSelection():
    def __init__(self, data, outcomes):
        self.data = data
        self.outcomes = outcomes
        self.rows, self.cols =self.data.shape

    # Pearson Correlation
    def pearson_selector(self, num_feats): # returns top num_feats correlated attributes indexes
        cors = {}
        for col in range(self.cols):
            data_subset = self.data[:, col]
            cor = abs(np.corrcoef(data_subset, self.outcomes)[0, 1])
            cors[col] = cor
        sorted_cor_idx = sorted(cors, key=cors.get, reverse=True)
        return sorted_cor_idx[:num_feats]

    def chi_squared(self, num_feats):
        # resulting_features = SelectKBest(chi2, k=num_feats).fit_transform(self.data, self.outcomes) # works but doesn't tell me which columns it's keeping
        chi_selector = SelectKBest(chi2, k=num_feats).fit(self.data, self.outcomes)
        selected_features = chi_selector.get_support() # get True/False result
        res = [i for i, val in enumerate(selected_features) if val] # return True value indeces
        return res

    def recursive_feature_elim(self, num_feats):
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=1, verbose=0)
        rfe_selector.fit(self.data, self.outcomes)
        selected_features = rfe_selector.get_support()
        res = [i for i, val in enumerate(selected_features) if val]
        return res

    # pretty sure this is doing the same things as rfe
    def lasso_selector(self, num_feats):
        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
        embeded_lr_selector.fit(self.data, self.outcomes)
        selected_features = embeded_lr_selector.get_support()
        res = [i for i, val in enumerate(selected_features) if val]
        return res
        
    def rf_selector(self, num_feats):
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
        embeded_rf_selector.fit(self.data, self.outcomes)
        selected_features = embeded_rf_selector.get_support()
        res = [i for i, val in enumerate(selected_features) if val]
        return res