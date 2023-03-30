#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alemayehu.taye
"""


from version_control_1 import *
RandomState = 2446

# importing and splitting the data 
DF = pd.read_csv(DatasetPath + "my_data.csv", low_memory=False)
Trn, Tst = train_test_split(DF,
                            test_size    = 0.2, 
                            random_state = RandomState)
