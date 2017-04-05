import pandas as pd
import numpy as np
import psycopg2
import os
from uuid import uuid4

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns


from memory_profiler import profile
import psycopg2
import pandas as pd
import numpy as np
import os
import psutil
import stat
from ConfigParser import ConfigParser
import logging
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, train_test_split
from sklearn.linear_model import *
from sklearn.neural_network import MLPClassifier
import sys
from scipy.sparse import csr_matrix
from scipy.io import mmwrite, mmread
from sklearn.externals import joblib
from sklearn.metrics import precision_score, roc_auc_score, make_scorer


############################## Now correlation / regression analysis
def age_bucket():
    if np.random.rand(1)[0] > 0.5:
        return '1'
    else:
        return '0'


def gender_bucket():
    if np.random.rand(1)[0] > 0.5:
        return '1'
    else:
        return '0'

def conv():
    if np.random.rand(1)[0] > 0.5:
        return 1
    else:
        return 0
###### Need to check the Shape of the data

data = pd.read_csv('filename.csv')

#
print data.shape



columns = 'user_id, age_bucket, primary_device, mc_affinity, gender, agweight, agrweight, agreweight, target, single_hh_flag, viewing_segment, tv_basead, tv_smad, fb_mobilead, fb_desktopad, tv_base_spend, tv_sm_spend, fb_mobile_spend, fb_desktop_spend, total_frequency, channel_reached, outcome_flag, conv, age_bucket2, bct_target, fb_target'.split(', ')

#### Channel reached may not be used together with the frequency per channel
categorical_attributes = ['age_bucket', 'age_bucket2', 'gender', 'target', 'bct_target', 'fb_target', 'single_hh_flag', 'viewing_segment', 'primary_device',
                          'channel_reached']

numerical_attributes = ['tv_basead', 'tv_smad', 'fb_mobilead', 'fb_desktopad']

# attributes to be used: age_bucket, age_bucket2, primary_device, gender, target, bct_target, fb_target, single_hh_flag, viewing_segment, tv_basead, tv_smad, fb_mobilead, fb_desktopad, channel_reached, outcome_flag, conv
data = pd.DataFrame(columns=columns)

n = 10000

data['user_id'] = [uuid4() for i in range(n)]

data.ix[:, 1:] = np.random.rand(n, len(columns)-1)


################# Define the buckets for each attributes, for reach data we don't need to do this
data['age_bucket'] = [age_bucket() for i in range(n)]

data['gender'] = [gender_bucket() for i in range(n)]

data['conv'] = [conv() for i in range(n)]
################# Define the buckets for each attributes, for reach data we don't need to do this



####### Descriptive analysis
data.groupby(['gender_bucket'])['user_id'].count()



### Transform the categorical data to dummy variables, need to change the columns
categorical_data = pd.get_dummies(data.ix[:, ['age_bucket', 'gender']], drop_first=True)

data_final = pd.concat([data[numerical_attributes], categorical_data], axis=1)

# plt.matshow(data_final.astype(float).corr())
# corr = data_final.astype(float).corr()
corr = data[numerical_attributes].astype(float).corr()

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=ax)


########## Linear Regression, Logistic or Decision Trees
########## It can be used for modeling the facebook target as well
Y = data['conv']
X = data_final[numerical_attributes + list(categorical_data.columns)]



##### Linear Regression
X_const = sm.add_constant(X, prepend=False)

mod = sm.OLS(Y, X_const)
res = mod.fit()
print res.summary()

with open('regression_result.txt', 'w') as reg_result:
    reg_result.write(str(res.summary()))


##### Logistic Regression
logit = LogisticRegression(C=1000)

logit.fit(X, Y)

print logit.coef_


##### Decision Trees

tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
tree.fit(X, Y)

tree.feature_importances_




def grid_search(n, X, y, base_estimator, params):
    grid_search_space = list(ParameterGrid(params))

    auc = [0] * len(grid_search_space)

    for i in range(len(grid_search_space)):
        logging.info('Parameters: ' + str(grid_search_space[i]))

        rf = base_estimator(**grid_search_space[i])

        kf = KFold(n_splits=n)

        auc_cv = []
        for train_index, test_index in kf.split(X):
            print train_index, test_index
            X_train, X_test = X.ix[train_index, :], X.ix[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            # It is important to train the ensemble of trees on a different subset
            # of the training data than the linear regression model to avoid
            # overfitting, in particular if the total number of leaves is
            # similar to the number of training samples

            X_train_rf, X_train_lr, y_train_rf, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

            # rf_enc = OneHotEncoder()
            # rf_lm = LogisticRegression()

            rf.fit(X_train_rf, y_train_rf)
            # rf_enc.fit(rf.apply(X_train_rf))
            #
            # rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

            # y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
            y_pred_rf_lm = rf.predict_proba(X_test)[:, 1]
            auc_cv.append(roc_auc_score(y_test, y_pred_rf_lm))

        auc[i] = np.mean(auc_cv)

        logging.info('AUC: ' + str(auc[i]))

    best_auc = max(auc)
    best_params = grid_search_space[auc.index(max(auc))]

    return best_auc, best_params


    # 1) Make the split,
    # 2) Run model, and keep track of the error rate
    # 3) average the error rate and find the parameter set that has the least error rate

def modeling(train, train_converted):
    logging.info('Modeling Starts')

    process = psutil.Process(os.getpid())
    logging.info('Current Memory Usage: ' + str(process.memory_info().rss / 1024.0 / 1024.0) + 'MB')

    models = {
        # 'LogisticRegression':
        #     {
        #         'estimator': LogisticRegression(penalty='l1'),
        #         'params': {'C': [0.001, 0.01, 0.1, 1], 'n_jobs': [-1]},
        #         'auc': 100000,
        #         'best_params': {}
        #     }
        # ,
        # 'ridge':
        #     {
        #         'estimator': Ridge(),
        #         'params': {'alpha': np.linspace(10e-3, 1, 100)},
        #         'test_error': [100000] * n,
        #         'test_error_mean': 100000
        #     },
        # 'svr':
        #     {
        #         'estimator': SVR(),
        #         'params': {'kernel': ('linear', 'rbf'), 'C': np.linspace(10e-2, 100, 100)},
        #         'test_error': [100000] * n,
        #         'test_error_mean': 100000,
        #         'test_total_pct_deviation': {}
        #     },
        'randomforest':
            {
                'estimator': RandomForestClassifier,
                'params': {'n_estimators': [100], 'min_samples_leaf': [5], 'n_jobs': [-1]},
                'best_auc': 0,
                'best_params': {}
            }
        ,

        # 'adaboost':
        #     {
        #         'estimator': AdaBoostClassifier,
        #         'params': {'n_estimators': [50, 300, 1000]},
        #         'best_auc': 0,
        #         'best_params': {}
        #     }
        # ,

        # 'mlp':
        #     {
        #         'estimator': MLPClassifier(),
        #         'params': {'activation': ['logistic'], 'hidden_layer_sizes': [(1000, )]},
        #         'auc': 100000,
        #         'best_params': {}
        #     }
        # ,
        # 'gradientboosting':
        #     {
        #         'estimator':  GradientBoostingClassifier(),
        #         'params': {'n_estimators': [10, 50], 'max_depth': [5, 10, 15]},
        #         'auc': 100000,
        #         'best_params': {}
        #     }


    }

    # for each model
    with open('result.txt', 'w') as output:
        for i in range(len(models.keys())):
            model = models.keys()[i]
            # define the estimator of the model
            estimator = models[model]['estimator']
            # define the parameters of the model
            params = models[model]['params']
            # pipe = Pipeline([('scale', StandardScaler(with_mean=False)), (model, estimator)])
            # grid search across parameters
            # auc_score = make_scorer(roc_auc_score)
            # grid_search = GridSearchCV(estimator=pipe, param_grid=params, scoring=auc_score)


            # fit the model from the grid search
            logging.info('Grid Search Starts')

            logging.info('Model: ' + model)
            logging.info('Params: ' + str(params))

            best_auc, best_params = grid_search(3, train, train_converted, estimator, params)

            # fit_model = grid_search.fit(train, train_converted)
            logging.info('Finished Grid Search')
            logging.info('Best AUC: ' + str(best_auc))
            logging.info('Best Params: ' + str(best_params))
            process = psutil.Process(os.getpid())
            logging.info('Current Memory Usage: ' + str(process.memory_info().rss / 1024.0 / 1024.0) + 'MB')

            # logging.info('Dump the model to file')
            # joblib.dump(fit_model, model + '.pkl', compress=1)
            # logging.info('Finished dumping the model')
            # process = psutil.Process(os.getpid())
            # logging.info('Current Memory Usage: ' + str(process.memory_info().rss / 1024.0 / 1024.0) + 'MB')
            # Error rate out from grid search
            # precision = fit_model.cv_results_[['mean_test_score', 'params']]
            #
            # print precision

            models[model]['best_auc'] = best_auc
            models[model]['best_params'] = best_params

            # output.write(model)
            # output.write("\nProportion of Class 1: " + str(np.mean(train_converted)))



        # models_rank = sorted(models.items(), key=lambda x: x[1]['test_error_mean'])
        models_rank = sorted(models.items(), key=lambda x: x[1]['best_auc'], reverse=True)

        # print [[item, models[item]['test_error_mean']] for item in models.keys()]
        logging.info(str([[item, models[item]['best_auc']] for item in models.keys()]))
        # print [[[advertiser, item, np.mean(models[item]['test_total_pct_deviation'][advertiser])] for advertiser in models[item]['test_total_pct_deviation'].keys()] for item in models.keys()]

        logging.info("\nBest Model is: " + models_rank[0][0])
        output.write("\nBest Model is: " + models_rank[0][0])
        output.write("\n" + str(models))
        logging.info("\nFinished Modeling Process")
        return models_rank[0][0]

modeling(X, Y)

def predicting(model, feature_selection):
    # demo_geo_list, test = create_sparse_matrix('predicting')
    logging.info('Prediction Process Starts')
    demo_geo_list, test, test_converted = create_sparse_matrix('predicting', feature_selection)

    process = psutil.Process(os.getpid())
    logging.info('Current Memory Usage: ' + str(process.memory_info().rss / 1024.0 / 1024.0) + 'MB')

    np.savez("test_x.npz", data=test.data, indices=test.indices, indptr=test.indptr, shape=test.shape)
    np.save("test_y.npy", test_converted)
    # test = np.load("hh_not_in_match.npz")
    model = joblib.loadafjadsjfa(model + '.pkl')

    logging.info('Start to make prediction')
    pred = model.predict(test)

    precision = precision_score(y_true=test_converted, y_pred=pred)

    logging.info("\nProportion of Class 1: " + str(np.mean(test_converted)))
    logging.info("\nPrecision: " + str(precision))
