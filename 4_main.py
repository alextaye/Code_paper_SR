#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alemayehu taye
"""

from 1_version_control import *
from 2_data import *
from 3_empirical_tuning import *

RandomState = 2446

best_params
# fit optimised RF and baseline OLS 
#-------------------------
param_grid_ = {"max_features":range(1,115), 
                "max_depth": range(1,40), 
                }

rf_ =     RandomForestRegressor(n_estimators = 500,
                                criterion    = 'mse', 
                                max_depth    = 
                                max_features = 
                                n_jobs       = -1,
                                random_state = RandomState,
                                verbose      = 1)

Pooled_opt = GridSearchCV(estimator          = rf_,
                          param_grid         = param_grid_, 
                          cv                 = 3, 
                          return_train_score = True, 
                          scoring            = "neg_root_mean_squared_error",
                            n_jobs             = -1,
                          verbose             = 1, 
                          refit              = True
                          ).fit(Trn.drop(columns = drop_), Trn['CBS_reason_mean'].ravel())

# Ols = LinearRegression().fit(Trn.drop(columns = drop_), Trn['CBS_reason_mean'].ravel())
