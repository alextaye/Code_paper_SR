#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alemayehu taye
"""

from 1_version_control import *
from 2_data import *
from 3_empirical_tuning import *

RandomState = 2446

# fit optimised RF and baseline OLS 
#-------------------------
rf_pooled =     RandomForestRegressor(n_estimators = 500,
                                criterion    = 'mse', 
                                max_depth    = best_params.max_depth, 
                                max_features = best_params.max_features
                                n_jobs       = -1,
                                random_state = RandomState,
                                      
                                verbose      = 1).fit(Trn[features], Trn['CBS_reason_mean'].ravel())

Ols_pooled = LinearRegression().fit(Trn[features], Trn['CBS_reason_mean'].ravel())

# Evaluate 
Evaluate(model = Ols, data = DF)
Evaluate(model = rf_pooled, data = DF)

# Post-training  
explainer_ = shap.TreeExplainer(rf_pooled) 
shap_values_ = explainer_.shap_values(Trn[features])  
shap_interaction_values = explainer_.shap_interaction_values(Trn[features])
permut_ = permutation_importance(rf_pooled,
                                  X = Trn[features], 
                                  y = Trn["CBS_reason_mean"],
                                  n_repeats    =  200,
                                  n_jobs       =  -1,
                                  random_state = RandomState)
 
  
