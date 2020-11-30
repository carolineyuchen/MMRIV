#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:58:08 2020

@author: afsaneh
"""
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import random
import scipy.sparse as sp
import matplotlib as mpl
import statistics 
from keras.models import Sequential
import tensorflow

import math
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import time


import itertools as it

from sklearn.linear_model import LinearRegression as LR
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessClassifier as GPC

##%matplotlib inline
plt.show()
sns.set_theme(font="tahoma", font_scale=0.6)


#%%
#param
N = [5,5000,10000,100000]
# [a,y,z,w] heat as a catalyst z temprature w 
a_XY = 0.5
a_XZ = 0.5

# [u,a,y,z,w]
m_e = [5, 0, 0, -1, 2]
# cov_e = [[1],[,1],[,,1],[,,,1]]
C = [1,1,2,1,4]

# U is a chi2 distribution
random.seed(100)
U = np.random.chisquare(m_e[0], N[1]).round(3)  # generates 5000 U's to 3.d.p.
U_inst = np.ones(N[1]).round(3)

# Z is noisy reading of U
random.seed(110)
eZ = np.random.normal(m_e[3], C[3], N[1])  # noise for Z
Z = (eZ - U).round(3)
Z_conU = (eZ - U_inst).round(3)  # TODO: what is this? constant U, or confounded by U?


random.seed(120)  # noise for W
eW = np.random.normal(m_e[4], C[4], N[1])
W = (eW + 2 * U).round(3)
W_conU = (eW + 2 * U_inst).round(3)


random.seed(130)
eX = np.random.normal(m_e[1], C[1], N[1])
X = (eX + a_XZ * Z + 2 * U ** 0.5).round(3)
X_conU = (eX + a_XZ * Z_conU + 2 * U_inst ** 0.5).round(3)

random.seed(19500)
eY = np.random.normal(m_e[2], C[2], N[1])
Y = (np.exp(a_XY * X ) + eY + -np.log10(U)).round(3)
Y_conU = (np.exp(a_XY * X_conU) + eY - np.log10(U_inst)).round(3)




D = pd.DataFrame([U,X,Y,Z,W]).T
D.columns=['U','X','Y','Z','W']
O=pd.DataFrame([X,Y,Z,W]).T
O.columns=['X','Y','Z','W']
D_conU=pd.DataFrame([U_inst,X_conU,Y_conU,Z_conU,W_conU]).T
D_conU.columns=['U','X','Y','Z','W']
