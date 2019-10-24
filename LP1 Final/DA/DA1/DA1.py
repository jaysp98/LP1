# -*- coding: utf-8 -*-
"""Lp1Da1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tXizTkYRF-pKQm670RfrNHExONdyCTbt
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
# %matplotlib inline


import matplotlib.pyplot as plt
import seaborn as sns

dat=pd.read_csv('Iris.csv')

dat[0:10]

dat.shape  #######how many features are there
list(dat.columns)

dat.dtypes  ##what are their types

dat['x1'].describe()    ###########statistics description for columns

dat['x2'].describe()

dat['x3'].describe()

dat['x4'].describe()

dat['class'].describe()

plt.hist(dat['x1'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()


plt.hist(dat['x2'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()


plt.hist(dat['x3'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()


plt.hist(dat['x4'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()

#######################box plot for single feature  same for rest

sns.boxplot(y=dat['x1'])

sns.boxplot(x='class',y=dat['x2'])

sns.boxplot(data=dat.ix[:,0:4])    #################for multiple

sns.boxplot(x=dat['class'],y=dat['x2'])  ############one vs all