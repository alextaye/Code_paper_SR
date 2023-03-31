#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alemayehu taye
"""

from version_control_1 import *
from data_2 import *
RandomState = 2446


with open(Global + "/features.txt", "r") as f:
    Text = f.read()
features = Text.split("\n")                                                                         # load feature lists

def cv_results(model):
    cv_results = pd.DataFrame(model.cv_results_)
    cv_results.loc[:, ["rank_test_score", "param_max_depth", "param_max_features", "mean_test_score", "std_test_score"]].\
        sort_values(by = "rank_test_score")
    cv_results["forest_depth"] =  model.best_params_["max_depth"]
    cv_results["forest_features"] = model.best_params_["max_features"]
    cv_results["best_score"] = model.best_score_
    cv_results.to_csv(path, mode = 'a', header = True)
    XX = cv_results[['param_max_depth', 'param_max_features', 'mean_test_score']]
    XX.loc[(XX['mean_test_score'] < 0), 'mean_test_score'] = XX.loc[(XX["mean_test_score"]>0), "mean_test_score"].min() + 0.0001
    CC = XX.pivot_table(index = "param_max_depth", columns = "param_max_features", values = "mean_test_score")
    return CC

# grid space 
param_grid_rf = {
                  "max_features": range(1, 35), 
                  "max_depth": range(1, 35)
                 }

rf_ =     RandomForestRegressor(n_estimators = 500,
                                criterion    = 'mse', 
                                n_jobs       = -1,
                                random_state = RandomState,
                                verbose      = 1)

clf_pooled = GridSearchCV(estimator          = rf_,
                          param_grid         = param_grid_rf, 
                          cv                 = 5, 
                          return_train_score = True, 
                          scoring            = "neg_root_mean_squared_error",
                          n_jobs             = -1,
                          verbose             = 1, 
                          refit              = True
                          ).fit(Trn[features], Trn['CBS_reason_mean'].ravel())

best_params = print("Best parameter:\t{}".format(clf_pooled.best_params_))
print("Best cv score:\t{:.6f}".format(clf_pooled.best_score_)) 
