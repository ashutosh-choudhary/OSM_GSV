import pickle
import constants as c
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import os
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix

PIPELINE_TYPE_TRAIN = "TRAIN"
PIPELINE_TYPE_TEST = "TEST"
seed = 7
validation_size = 0.20
num_folds = 10
scoring = 'accuracy'

class svm_coh:

    def __init__(self):
        self

    def hyperparameter_optimization_rf_coh(self, X, Y):
        """
        Hyperparameter optimization for rf - coh features
        :param X: Features - training data
        :param Y: Labels - training data
        :return:
        """
        num_instances = len(X)
        print "\n\nInside Hyperparameter Optimization: "

        #parameters = {'kernel':['rbf', 'linear'], 'C': [10 ** x for x in range(-3, 4)], 'gamma': [10 ** x for x in range(-3, 4)]}
        parameters = {'n_estimators':range(200,300,20), 'max_features': range(1,5), 'max_depth': range(10, 40, 10)}
        grid = GridSearchCV(RandomForestClassifier(), param_grid=parameters, scoring=scoring, cv=5, verbose=5)
        grid.fit(X, Y)

        print "================================================"
        print("The best parameters after first run are %s with a score of %0.6f"
              % (grid.best_params_, grid.best_score_))
        best_score_coarse = grid.best_score_
        best_param_coarse = grid.best_params_

        # Save the clf
        DIR = c.PATH_MODEL_HOME
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        filename_grid = DIR + c.FILENAME_BEST_RF_COARSE
        print 'rf_coh: ', 'saving file ...', filename_grid
        joblib.dump(grid, filename_grid)

    def hyperparameter_optimization_ab_coh(self, X, Y):
        """
        Complete processing of Bank Queue dataset including
        feature selection and hyper-parameter optimization
        :param X: Features - training data
        :param Y: Labels - training data
        :return:
        """
        best_score_total = -float('inf')
        best_model_total = ""
        best_params_total = {}

        # MLC Perceptron with selected features
        nn = AdaBoostClassifier()

        # NN with selected features
        parameters = {'n_estimators': range(10,150,10)}
        grid = GridSearchCV(nn, param_grid=parameters, scoring=scoring, cv=5, verbose=5)
        grid.fit(X, Y)
        print "======================================================="
        print("The best parameters for AB is %s with a score of %0.6f"
              % (grid.best_params_, grid.best_score_))

        # Save the clf
        DIR = c.PATH_MODEL_HOME
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        filename_grid = DIR + c.FILENAME_BEST_AB_COARSE
        print 'nn_gist: ', 'saving file ...', filename_grid
        joblib.dump(grid, filename_grid)

        if grid.best_score_ > best_score_total:
            best_score_total = grid.best_score_
            best_params_total = grid.best_params_

    def load_clf_rf(self):
        DIR = c.PATH_MODEL_HOME
        filename_grid = DIR + c.FILENAME_BEST_RF_COARSE
        print 'rf_Coh: ', 'found file ...', filename_grid
        grid = joblib.load(filename_grid)
        return grid

    def load_clf_ab(self):
        DIR = c.PATH_MODEL_HOME
        filename_grid = DIR + c.FILENAME_BEST_AB_COARSE
        print 'ab_coh: ', 'found file ...', filename_grid
        grid = joblib.load(filename_grid)
        return grid

    def final_test_rf(self):
        X, y, XT, yT = self.load_data()
        rf = self.load_clf_rf()
        y_pred = rf.best_estimator_.predict(XT)
        score = rf.best_estimator_.score(XT, yT)
        cnf_metrix = confusion_matrix(y_pred, yT)
        print 'Random Forest'
        print "Score: ", score
        print "cnf_metrix: ", cnf_metrix

    def final_test_ab(self):
        X, y, XT, yT = self.load_data()
        ab = self.load_clf_ab()
        y_pred = ab.best_estimator_.predict(XT)
        score = ab.best_estimator_.score(XT, yT)
        cnf_metrix = confusion_matrix(y_pred, yT)
        print 'AdaBoost Classifier'
        print "Score: ", score
        print "cnf_metrix: ", cnf_metrix

    def load_data(self):

        # Load the Data
        filename_train = c.DATA_TRAINING_COH_PICKLED
        filename_test = c.DATA_TEST_COH_PICKLED
        train_data = pickle.load(open(filename_train, "rb"))
        test_data = pickle.load(open(filename_test, "rb"))
        print "\n\n", "model_estimation: base_main: ", "Loaded pickled data successfully"
        features_train = np.array(train_data)[:, :-1]
        labels_train = np.array(train_data)[:, -1].astype(int)
        features_test = np.array(test_data)[:, :-1]
        labels_test = np.array(test_data)[:, -1].astype(int)
        # print "model_estimation: base_main: ", ":", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape
        n_features_train = preprocessing.scale(features_train)
        n_features_test = preprocessing.scale(features_test)

        return n_features_train, labels_train, n_features_test, labels_test
        #return n_features_train[:100], labels_train[:100], n_features_test[:100], labels_test[:100]