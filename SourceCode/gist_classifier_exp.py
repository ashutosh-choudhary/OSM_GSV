import pickle
import constants as c
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
import os
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix

PIPELINE_TYPE_TRAIN = "TRAIN"
PIPELINE_TYPE_TEST = "TEST"
seed = 7
validation_size = 0.20
num_folds = 10
scoring = 'accuracy'

class svm_gist:

    def __init__(self):
        self


    def hyperparameter_optimization_svm_gist(self, X, Y):
        """
        Hyperparameter optimization for svm - gist features
        :param X: Features - training data
        :param Y: Labels - training data
        :return:
        """
        num_instances = len(X)
        print "\n\nInside Hyperparameter Optimization: "

        #parameters = {'kernel':['rbf', 'linear'], 'C': [10 ** x for x in range(-3, 4)], 'gamma': [10 ** x for x in range(-3, 4)]}
        parameters = {'kernel':['rbf', 'linear'], 'C': [10 ** x for x in range(-2, 2)], 'gamma': [10 ** x for x in range(-2, 2)]}
        grid = GridSearchCV(SVC(), param_grid=parameters, scoring=scoring, cv=5, verbose=5)
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

    #    filename_grid = DIR + c.FILENAME_BEST_SVM_COARSE
    #    print 'svm_gist: ', 'saving file ...', filename_grid
    #    joblib.dump(grid, filename_grid)

        parameters_fine = {'kernel':['rbf', 'linear'], 'C': np.ndarray.tolist(
            np.logspace(np.log10(grid.best_params_['C']) - 1, np.log10(grid.best_params_['C']) + 1, 5)),
                           'gamma': np.ndarray.tolist(np.logspace(np.log10(grid.best_params_['gamma']) - 1,
                                                                  np.log10(grid.best_params_['gamma']) + 1, 5))}
        grid_fine = GridSearchCV(SVC(), param_grid=parameters_fine, scoring=scoring, cv=5, verbose=5)
        grid_fine.fit(X, Y)

        print "================================================"
        print("The best parameters after fine tuning are %s with a score of %0.6f"
              % (grid_fine.best_params_, grid_fine.best_score_))

        # Save the clf
        DIR = c.PATH_MODEL_HOME
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        filename_grid = DIR + c.FILENAME_BEST_SVM_COARSE
        print 'svm_gist: ', 'saving file ...', filename_grid
        joblib.dump(grid, filename_grid)
        filename_fine = DIR + c.FILENAME_BEST_SVM_FINE
        print 'svm_gist: ', 'saving file ...', filename_fine
        joblib.dump(grid, filename_fine)

    def hyperparameter_optimization_mlcp_gist(self, X, Y):
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
        nn = MLPClassifier(alpha=.01)

        # NN with selected features
        parameters = {'hidden_layer_sizes': [25, 50, 100, 150]}
        grid = GridSearchCV(nn, param_grid=parameters, scoring=scoring, cv=5, verbose=5)
        grid.fit(X, Y)
        print "======================================================="
        print("The best parameters for NN is %s with a score of %0.6f"
              % (grid.best_params_, grid.best_score_))

        # Save the clf
        DIR = c.PATH_MODEL_HOME
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        filename_grid = DIR + c.FILENAME_BEST_NN_COARSE
        print 'nn_gist: ', 'saving file ...', filename_grid
        joblib.dump(grid, filename_grid)

        if grid.best_score_ > best_score_total:
            best_score_total = grid.best_score_
            best_params_total = grid.best_params_

    def final_test_svm(self):
        X, y, XT, yT = self.load_data()
        svm = self.load_clf_svm()
        y_pred = svm.best_estimator_.predict(XT)
        score = svm.best_estimator_.score(XT, yT)
        cnf_metrix = confusion_matrix(y_pred, yT)
        print "Score: ", score
        print "cnf_metrix: ", cnf_metrix

    def final_test_nn(self):
        X, y, XT, yT = self.load_data()
        nn = self.load_clf_nn()
        y_pred = nn.best_estimator_.predict(XT)
        score = nn.best_estimator_.score(XT, yT)
        cnf_metrix = confusion_matrix(y_pred, yT)
        print 'MLP Perceptron'
        print "Score: ", score
        print "cnf_metrix: ", cnf_metrix

    def load_clf_svm(self):
        DIR = c.PATH_MODEL_HOME
        filename_grid = DIR + c.FILENAME_BEST_SVM_COARSE
        print 'svm_gist: ', 'found file ...', filename_grid
        grid = joblib.load(filename_grid)
        best_score_coarse = grid.best_score_
        best_param_coarse = grid.best_params_

        filename_fine = DIR + c.FILENAME_BEST_SVM_FINE
        print 'svm_gist: ', 'found file ...', filename_fine
        grid_fine = joblib.load(filename_fine)

        if grid_fine.best_score_ > best_score_coarse:
            print "\n Found a better fine tunes parameter set:"
            best_grid = grid_fine
            best_score = grid_fine.best_score_
            best_params = grid_fine.best_params_
        else:
            best_grid = grid
            best_score = best_score_coarse
            best_params = best_param_coarse
        print("The final best parameters are %s with a score of %0.6f" % (best_params, best_score))
        return best_grid

    def load_clf_nn(self):
        DIR = c.PATH_MODEL_HOME
        filename_grid = DIR + c.FILENAME_BEST_NN_COARSE
        print 'nn_gist: ', 'found file ...', filename_grid
        grid = joblib.load(filename_grid)
        return grid

    def load_data(self):

        # Load the Data
        filename_train = c.DATA_TRAINING_GIST_PICKLED
        filename_test = c.DATA_TEST_GIST_PICKLED
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

