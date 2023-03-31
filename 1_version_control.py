#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alemayehu taye
"""
import pandas as pd
import numpy as np
import scipy as sp
import time
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#### Evaluation metrics
from sklearn.metrics import r2_score, 
from sklearn.metrics import mean_absolute_error, 
from sklearn.metrics import mean_squared_error, 

#### Setting Console desplay 
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

#### make path
currentDir = os.getcwd()
Global = ""
DatasetPath = " "
ResultPath = ""

#### Define important functions

#### 1)
def Evaluate(model, data): 
     Trn, Tst        = train_test_split(data,
                                     test_size    = 0.2, 
                                     random_state = RandomState)
     y = "CBS_reason_mean"
     X = list(data.drop(columns = drop))
     index = ["R2", "RMSE", "MAE", "MAPE"]
     Prediction_test  = model.predict(Tst[X])
     Prediction_train = model.predict(Trn[X])
     Rmse_test  =  np.sqrt(mean_squared_error(Tst[y], Prediction_test)).round(3)
     Rmse_train =  np.sqrt(mean_squared_error(Trn[y], Prediction_train)).round(3)
     Mae_test  = mean_absolute_error(Tst[y], Prediction_test).round(3)
     Mae_train = mean_absolute_error(Trn[y], Prediction_train).round(3)
     stats_test = [Rmse_test, Mae_test]
     stats_train = [Rmse_train, Mae_train] 
     return pd.DataFrame({"Testset": stats_test, "Trainset": stats_train}, index =index).T

#### 2)
def shap_sum_plot(shap_values_, DF, save_name, max_feature = 25, figsize = (15, 15)):
  ## shap_values_ takes SHAP values, DF takes dataset used to compute the SHAP, save_name takes string name.
    plt.close()
    my_cmap = plt.get_cmap('coolwarm') # seismic, viridis, bwr, coolwarm, magma
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axp = shap.summary_plot(shap_values_, 
                      DF, 
                      plot_size = figsize,
                      max_display = max_feature,
                      cmap = my_cmap,
                      color_bar = False,
                      use_log_scale = False,
                      plot_type = "dot", # dot, bar, violin, compact_dot"
                      show = False)
    cb = plt.colorbar(axp, ax = [ax], ticks = [0.0, 9.0], location = 'left', shrink = .35, pad = 0.1, aspect = 20)
    cb.ax.set_yticklabels(['Low', 'High'], fontsize=16, weight = "normal")
    cb.set_label('Feature value', fontsize = 20, color = 'k', weight = 'normal')
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(axis="both", colors ="k", labelsize = 20)
    plt.xlabel('SHAP value (impact on model output)', fontsize=22, color = 'k')
    print (cb)
    plt.grid(True, linestyle = ":", linewidth = 0.5)
    plt.xticks(np.arange(-10000, 25000, 5000)) # force xticks in plt
    plt.savefig(ResultPath + f'{save_name}.pdf', 
            pad_inches = 0, 
            orientation = 'portrait', 
            bbox_inches="tight")
    
    #### 3)
    def Shap_corr(shap_values_, DF, max_feature, figsize, save_name):
    shap_vvv = pd.DataFrame(shap_values_)
    dff      = DF
    FFeature_list       = dff.columns
    shap_vvv.columns    = FFeature_list
    df_v        = dff.copy().reset_index().drop('index',axis=1)
    corr_list_  = list()
    corr_coef = list()
    p_value = list()
    for i in FFeature_list:
        b = np.corrcoef(shap_vvv[i],df_v[i])[1][0]
        corr_list_.append(b)
    # point-biserial correlation 
    stat, p = stats.pointbiserialr(x = df_v[i], y = shap_vvv[i])
    p_value.append(p)
    corr_coef.append(stat)
    biserial_corr_df = pd.DataFrame({"Correlation": corr_coef,"P-value": ["%.5f" % l for l in p_value]}, 
                                index = FFeature_list)
    corr_df_            = pd.concat([pd.Series(FFeature_list),pd.Series(corr_list_)],axis=1).fillna(0)
    corr_df_.columns    = ['Variable','Corr']
    shap_abs_   = np.abs(shap_vvv)
    k_          = pd.DataFrame(shap_abs_.mean()).reset_index()
    k_.columns  = ['Variable','SHAP_abs']
    k2_         = k_.merge(corr_df_,left_on = 'Variable',right_on='Variable',how='inner')
    k2_ = k2_.sort_values(by='SHAP_abs', ascending = False)
    k2_ = k2_.reset_index()                          
    df = k2_[0:max_feature:].iloc[::-1]
    norm = plt.Normalize(vmin = min(df.Corr), vmax = max(df.Corr), clip=True)
    mapper = mpl.cm.ScalarMappable(norm = norm, cmap = "coolwarm") # seismic, bwr, viridis, coolwarm, magma
    colors = [mapper.to_rgba(x) for x in df.Corr]  
    labels = list(df.Variable)
    fig, ax = plt.subplots(figsize=figsize)  
    ax.barh(y = df.Variable, left = False, width = df.SHAP_abs, color = colors)  
    ax.tick_params(axis="both", colors="k", labelsize = 22)
    ax.set_yticklabels([])
    for i, yi in enumerate(list(df.Variable)):
        ax.text(-.027, yi, labels[i], weight = 'bold', fontsize = 22, horizontalalignment='center', verticalalignment='center')
    ax.set(ylabel=None)
    mapper._A = []          
    cbar = plt.colorbar(mapper, orientation="vertical", shrink = .35, pad = 0.1, aspect = 20) #orientation="horizontal" location = 'left', 
    cbar.set_label('Direction of impact', fontsize = 20, color = 'k', weight = 'normal')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='k', fontsize=20, weight = 'normal')
    plt.setp(ax.patches, linewidth=.75)
    plt.margins(y = 0.02) # rmove space form above and below
    plt.grid(True, linestyle = ":", linewidth = .5)
    plt.xlabel("Mean|SHAP Value|", fontsize=22, color = 'k')
    ToSee = k2_[0:max_feature:].drop(columns = "index")
    print(list(ToSee.Variable))
    plt.savefig(ResultPath + f'{save_name}.pdf', pad_inches = 0, orientation = 'portrait', bbox_inches="tight")   
    return ax
