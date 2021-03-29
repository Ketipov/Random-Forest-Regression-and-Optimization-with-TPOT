#!/usr/bin/env python
# coding: utf-8
# Random Forest - example Agreeableness (independent_var) and Product descriptions (dependent_var)


# Import the libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib



# Import dataset
df = pd.read_csv(r"your directory")
#print(df)
#df.describe()




# Assigning the input and output values resp. dividing data into attributes and labels

y = df.iloc[0:, 6].values    # Product descriptions (dep_var)
X = df.iloc[0:, 40].values   # Agreeableness (indep_var)


y = np.array(y).astype('float')
X = np.array(X).astype('float')
X = X.reshape(-1,1)



# Divide the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)




# Create the regressor object
    # The n_estimators parameter defines the number of trees in the random forest
regressor = RandomForestRegressor(n_estimators=150, random_state=0)

# Fit the regressor with X and y data 
regressor.fit(X_train, y_train)                                 

y_pred = regressor.predict(X_test)



# Check predicted values
y_pred


#Plot
plt.plot(y_test, label='Actual')
plt.plot(y_pred, color="red", label='Predicted')
plt.legend()
plt.show()



# Evaluate the algorithm using MAE, RMSE, and MAPE (resp. Accuracy(100% - MAPE) for better interpretation)
import sklearn.metrics as metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_test - y_pred)
mape = 100 * (errors / y_test)

# Calculate and display accuracy (100% - MAPE)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Calculate and print MAPE again 
def MAPE(y_pred, y_test):
    return ( abs((y_test - y_pred) / y_test).mean()) * 100
print ('My MAPE: ' + str(MAPE(y_pred, y_test)) + ' %' )




# Optimization with GridSearchCV -
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# Import GridSearchCV method from sklearn library which is able to obtain the best parameters for the algorithm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

# Find the best parameters for the model
parameters = {
    'max_depth': [20, 50, 80, 100],
    'n_estimators': [150, 175, 200, 400]
}
gridforest = GridSearchCV(regressor, parameters, cv = 10, n_jobs = -1, verbose = 1)
gridforest.fit(X_train, y_train)
gridforest.best_params_

                 # 10 folds for each of 16 candidates = total of 160 fits




hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

                        # bootstrap = method for sampling data points (with or without replacement)
                                # default=True (if False, the whole dataset is used to build each tree)

# Create a pipeline for the cross validation       
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=150))

clf = GridSearchCV(pipeline, hyperparameters, n_jobs = -1, verbose = 2, cv=10) 
                # cv - determines the cross-validation splitting strategy 
                       # none - to use the default 5-fold cross validation

# Fit the grid search to the data
clf.fit(X_train, y_train)
clf.best_params_

y_pred_1 = clf.predict(X_test)

# New scores
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_1))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_1)))

errors = abs(y_test - y_pred_1)  
mape = 100 * (errors / y_test)   
# Calculate and display accuracy
accuracy_1 = 100 - np.mean(mape)
print('Accuracy_1:', round(accuracy_1, 2), '%.')

print('Improvement of {:0.2f}%.'.format( 100 * (accuracy_1 - accuracy) / accuracy))

                # 10 folds for each of 12 candidates = total of 120 fits


# Save model for future using joblib.dump

#Check the values
y_pred_1


#Plot
plt.plot(y_test, label='Актуални стойности')
plt.plot(y_pred_1, color="red", label='Прогноза')
plt.legend()
plt.show()



# Otpimization with TPOT - a Python Automated Machine Learning tool that optimizes machine learning pipelines 
                                                            # using Genetic Programming
# http://epistasislab.github.io/tpot/api/#regression

from tpot import TPOTRegressor
from sklearn.metrics import make_scorer


tpot = TPOTRegressor(generations=10, population_size=68, random_state=101,
                     verbosity=2,
                     cv=10,
                     n_jobs=-1)

# The total number of pipelines is equal to POPULATION_SIZE + (GENERATIONS x OFFSPRING_SIZE), whereat
         # offsping_size is to the number of population_size (if not defined)

# Important Parameters:
        # generations - int or None, default=100,
                # number of iterations to the run pipeline optimization process
        # population_size - int, default=100,
                # number of individuals to retain in the genetic programming population every generation
        # offspring_size - number of offspring to produce in each genetic programming generation
                # by default, the number of offspring is equal to the number of population size
        # cv - must be int, cross-validation generator, or an iterable, optional (default=5)
        # n_jobs - must be int, optional (default=1)
               # number of processes to use in parallel for evaluating pipelines during the TPOT optimization process
               #  n_jobs=-1 will use as many cores as available on the computer
        # random_state must be integer or None, default=None, to be used in order to be sure that TPOT will give 
               # the same results each time against the same data set with that seed
        # warm_start - boolean, default=False
               # to indicate whether the TPOT instance will reuse the population from previous calls to fit()
        # early_stop - integer, default: None,
              # how many generations TPOT checks whether there is no improvement in optimization process, it ends
              # the optimization process if there is no improvement
        # verbosity - integer, default=0,
              # how much information TPOT communicates while it's running, verbosity=2 means 
                               # TPOT will print more information and provide a progress bar
                               # verbosity=3 means TPOT will print everything and provide a progress bar
        # max_time_mins - integer or None, default=None,
              # it defines how many minutes TPOT has to optimize the pipeline

            
# Start a timer
import time
start = time.time()


tpot.fit(X_train,y_train)
tpot.export('TPOT_RF_Pers_E_Shopping.py')

results = tpot.predict(X_test)
y_pred_GP = results 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_GP))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_GP))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_GP)))

# New score
errors = abs(y_test - y_pred_GP)
mape = 100 * (errors / y_test)
# Calculate and display accuracy_tpot
accuracy_tpot = 100 - np.mean(mape)
print('Accuracy_tpot:', round(accuracy_tpot, 2), '%.')

print('Improvement of Accuracy with TPOT_Regression of: {:0.2f}%.'.format( 100 * (accuracy_tpot - accuracy) / accuracy))


# End the timer and print execution time in seconds
stop = time.time()
print("The time of the run:", stop - start)



#Check the values
y_pred_GP


#Plot
plt.plot(y_test, label='Actual')
plt.plot(y_pred_GP, color="red", label='Predicted')
plt.legend()
plt.show()




