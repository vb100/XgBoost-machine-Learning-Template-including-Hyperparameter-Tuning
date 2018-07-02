# -*- coding: utf-8 -*-
# Demo from: https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html
# Coded by Vytautas Bielinskas
# Date: 2018-06-20

# ::: Load dataset with Pandas :::
import pandas as pd
file = 'Dataset/Training/Features_Variant_1.csv'
df = pd.read_csv(file, header = None)
df.sample(n = 5)

# ::: Check the size of the dataset :::
print('Dataset has {} entries and {} features.'.format(*df.shape))

''' In order to evaluate the performance of our model, we need to train it on a 
sample of the data and test it on an other. We can do this easily with the 
function train_test_split from scikit-learn. First, let's extract the features 
and the target from our dataset.'''

# ::: Extract the features and the target from our dataset :::
X, y = df.loc[:, :52].values, df.loc[:, 53].values

# ::: Keep 90% of the dataset for training, and 10% for testing :::
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# ::: Loading data into DMatrices :::
''' In order to use the native API for XGBoost, we need to build DMatrices.'''
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

# ::: Building a baseline model :::
'''We are going to use mean absolute error (MAE) to evaluate the quality of our
 predictions. MAE is a common and simple metric that has the advantage of being
 in the same unit as our target, which means it can be compared to target values
 and easily interpreted.'''
from sklearn.metrics import mean_absolute_error

'''For our baseline, we will keep things simple and predict that each new post 
will get the mean number of comments that we observed in the training set.'''

import numpy as np
# "Learn" the mean from the training data
mean_train = np.mean(y_train)

# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train

# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)

print('Baseline MAE is {:.2f}.'.format(mae_baseline))
'''That is, the prediction is, on average, 11.31 comments off from the actual 
number of comments a post receives. Is that good? Well not really, if you look 
at the mean we just computed, you will see that the average number of comments 
for a post in the training set is a bit more than 7.'''

#------------------------------------------------------------------------------
# Training and Tuning an XGBoost model
'''Here we will tune 6 of the hyperparameters that are usually having a big 
impact on performance.'''

params = {
        'max_depth' : 6,
        'min_child_weight' : 1,
        'eta' : .3,
        'subsample' : 1,
        'colsample_bytree' : 1,
        # Other parameters
        'objective' : 'reg:linear'
        }

''' The first parameter we will look at is not part of the params dictionary, 
but will be passed as a standalone argument to the training method. This 
parameter is called num_boost_round and corresponds to the number of boosting 
rounds or trees to build. '''

# ::: Evaluation metric we are interested in to our params dictionary :::
params['eval_metric'] = 'mae'

num_boost_round = 999

model = xgb.train(
        params,
        dtrain,
        num_boost_round = num_boost_round,
        evals = [(dtest, 'Test')],
        early_stopping_rounds = 10
        )

print('Best MAE: {:.2f} with {} rounds.'.format(
        model.best_score,
        model.best_iteration+1))

# ::: Using XGBoost CV
'''In order to tune the other hyperparameters, we will use the cv function from
 XGBoost. It allows us to run cross-validation on our training dataset and 
 returns a mean MAE score.'''

cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round = num_boost_round,
        seed = 42,
        nfold = 5,
        metrics = ['mae'],
        early_stopping_rounds = 10
        )

cv_results
'''cv returns a table where the rows correspond to the number of boosting trees
 used, here again, we stopped before the 999 rounds (fortunately!).'''
 
''' we will only try to improve the mean test MAE. We can get the best MAE 
score from cv with:'''
cv_results['test-mae-mean'].min()

# You can try wider intervals with a larger step between
# each value and then narrow it down. Here after several
# iteration I found that the optimal value was in the
# following ranges.

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]

# ::: Cross validation
# Define initial best params and MAE
min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))

    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )

    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

'''We get the best score with a max_depth of 10 and min_child_weight of 6, so 
let's update our params'''

params['max_depth'] = 10
params['min_child_weight'] = 6

# :::
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_mae = float("Inf")
best_params = None

# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))

    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )

    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

params['subsample'] = .8
params['colsample_bytree'] = 1.

# ::: ETA parameter
%time

min_mae = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))

    # We update our parameters
    params['eta'] = eta

    # Run and time CV
    %time cv_results = xgb.cv(params, 
                              dtrain,
                              num_boost_round=num_boost_round,
                              seed=42,
                              nfold=5,
                              metrics=['mae'],
                              early_stopping_rounds=10)

    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta

print("Best params: {}, MAE: {}".format(best_params, min_mae))