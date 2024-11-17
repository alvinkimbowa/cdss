#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def get_sepsis_score(data, model):
    x_mean = np.array([ 22.48,  83.57,  97.24,  36.92, 123.13,  81.84,  62.91,  18.25,  -0.46,   0.5,
                       7.39,  40.32,  20.48,   7.82,   1.34, 131.07,   2.04,   4.14,  31.47, 10.58, 
                       11.46, 196.31, 62.25])
    x_std = np.array([ 15.22, 16.51,  2.88,  0.71, 22.19, 15.71, 13.51,  4.95,  3.39,  0.18,
                      0.06,  7.28, 16.33,  2.13,  1.57, 42.49,  0.37,  0.56,  5.34,  1.87,  
                      5.37, 93.23, 15.82])

    # impute missing values with forward fill
    for i in range(1, len(data)):
        mask = np.isnan(data[i])
        data[i][mask] = data[i - 1][mask]

    x = data[-1, 0:23]  # use only the current time step for prediction
    x_norm = np.nan_to_num((x - x_mean) / x_std)    # normalize the input with training set mean and std

    # logistic regression model
    const = [-4.0435]
    coeffs = np.array([ 0.1914,  0.1191, -0.0031,  0.3432,  0.0792, -0.1973, -0.0089,  0.0857,  0.0224, 0.2323,
                        -0.002,   0.0038,  0.1466, -0.0475,  0.0732,  0.0784, -0.043,  -0.169, -0.1041, 0.096,
                        0.0938,  0.0728, -0.0367])
    z = const + np.dot(x_norm, coeffs)
    score = 1.0 / (1 + np.exp(-z))  # sigmoid
    score = min(max(score,0),1)    # clamp to [0,1] just in case of numerical instability

    # threshold for sepsis prediction
    thresh = 0.0287
    label = score > thresh

    return score, label

def load_sepsis_model():
    return None
