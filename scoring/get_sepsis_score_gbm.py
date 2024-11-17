#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import lightgbm as lgb

def get_sepsis_score(data, model):
    x_mean = np.array([ 22.48,  83.57,  97.24,  36.92, 123.13,  81.84,  62.91,  18.25,  -0.46,   0.5,
                       7.39,  40.32,  20.48,   7.82,   1.34, 131.07,   2.04,   4.14,  31.47, 10.58, 
                       11.46, 196.31, 62.25])
    x_std = np.array([ 15.22, 16.51,  2.88,  0.71, 22.19, 15.71, 13.51,  4.95,  3.39,  0.18,
                      0.06,  7.28, 16.33,  2.13,  1.57, 42.49,  0.37,  0.56,  5.34,  1.87,  
                      5.37, 93.23, 15.82])

    for i in range(1, len(data)):
        mask = np.isnan(data[i])
        data[i][mask] = data[i - 1][mask]
    
    x = data[-1, 0:23]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    x_norm = np.array(x_norm)
    x_norm = x_norm.reshape(-1,23)
    score=model.predict(x_norm)
    score=min(max(score,0),1)
    label = score > 0.0282

    return score, label

def load_sepsis_model():
    return lgb.Booster(model_file='lightgbm_python.model')
