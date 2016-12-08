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

        parameters = {'kernel':['rbf', 'linear'], 'C': [10 ** x for x in range(-3, 4)], 'gamma': [10 ** x for x in range(-3, 4)]}
        # parameters = {'C': [10 ** x for x in range(0, 2)], 'gamma': [10 ** x for x in range(-2, 2)]}
        grid = GridSearchCV(SVC(), param_grid=parameters, scoring=scoring, cv=10, verbose=5)
        grid.fit(X, Y)

        print "================================================"
        print("The best parameters after first run are %s with a score of %0.6f"
              % (grid.best_params_, grid.best_score_))
        best_score_coarse = grid.best_score_
        best_param_coarse = grid.best_params_

        parameters_fine = {'kernel':['rbf', 'linear'], 'C': np.ndarray.tolist(
            np.logspace(np.log10(grid.best_params_['C']) - 1, np.log10(grid.best_params_['C']) + 1, 5)),
                           'gamma': np.ndarray.tolist(np.logspace(np.log10(grid.best_params_['gamma']) - 1,
                                                                  np.log10(grid.best_params_['gamma']) + 1, 5))}
        grid_fine = GridSearchCV(SVC(), param_grid=parameters_fine, scoring=scoring, cv=10, verbose=5)
        grid_fine.fit(X, Y)

        print "================================================"
        print("The best parameters after fine tuning are %s with a score of %0.6f"
              % (grid_fine.best_params_, grid_fine.best_score_))

        # Save the clf
        DIR = c.PATH_RESOURCES_HOME + 'results/'
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        filename_grid = DIR + 'best_svm_gist_coarse.pkl'
        print 'svm_gist: ', 'saving file ...', filename_grid
        joblib.dump(grid, filename_grid)
        filename_fine = DIR + 'best_svm_gist_fine.pkl'
        print 'svm_gist: ', 'saving file ...', filename_fine
        joblib.dump(grid, filename_fine)


        #print "\nStart Plotting Coarse hyperparameter plot..."
        #scores = [(-1 * x[1]) for x in grid.grid_scores_]
        #scores = np.array(scores).reshape(len(parameters['C']), len(parameters['gamma']))
        #
        #for ind, i in enumerate(parameters['C']):
        #    plt.plot(np.log10(parameters['gamma']), scores[ind], label='C: ' + str(i))
        #
        #plt.title('Robotic Arm SVR Hyperparameters optimization')
        #plt.legend()
        #plt.xlabel('Gamma')
        #plt.ylabel('Mean Error')
        #plt.savefig('../Figures/Robotic_Arm_SVR_Hyperparameter.png')
        #plt.gcf().clear()
        #
        #print "\nStart Plotting Fine tuning hyperparameter plot..."
        #scores = [(-1 * x[1]) for x in grid_fine.grid_scores_]
        #scores = np.array(scores).reshape(len(parameters_fine['C']), len(parameters_fine['gamma']))
        #
        #for ind, i in enumerate(parameters_fine['C']):
        #    plt.plot(np.log10(parameters_fine['gamma']), scores[ind], label='C: ' + str(i))
        #
        #plt.title('Robotic Arm SVR Hyperparameters optimization - Fine Tuning')
        #plt.legend()
        #plt.xlabel('Gamma')
        #plt.ylabel('Mean Error')
        #plt.savefig('../Figures/Robotic_Arm_SVR_Hyperparameter_Fine.png')
        #plt.gcf().clear()

    def hyperparameter_optimization_with_feature_selection(self, X, Y):
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

        # KNN with selected features
        nn = MLPClassifier(hidden_layer_sizes=50,alpha=.01)
        # SVc with selected features
        svc = SVC()

        # NN with selected features
        parameters = {'hidden_layer_sizes': [25, 50, 100, 150]}
        grid = GridSearchCV(nn, param_grid=parameters, scoring=scoring, cv=5, verbose=5)
        grid.fit(X, Y)
        print "======================================================="
        print("The best parameters for NN is %s with a score of %0.6f"
              % (grid.best_params_, grid.best_score_))

        # Save the clf
        DIR = c.PATH_RESOURCES_HOME + 'results/'
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        filename_grid = DIR + 'best_nn_gist_coarse.pkl'
        print 'nn_gist: ', 'saving file ...', filename_grid
        joblib.dump(grid, filename_grid)

        if grid.best_score_ > best_score_total:
            best_score_total = grid.best_score_
            best_params_total = grid.best_params_

        #for i in range(2, 10): #X.shape[1]
        #    sel = SelectKBest(f_classif(X, Y), k=i)
        #    sel.fit(X, Y)
        #    new_feature_set = sel.transform(X)
        #    print "Exp Feature set: ", new_feature_set.shape
        #
        #    # SVM
        #    parameters = {'kernel':['rbf', 'linear'], 'C': [10 ** x for x in range(0, 2)], 'gamma': [10 ** x for x in range(-2, 2)]}
        #    grid = GridSearchCV(SVC(), param_grid=parameters, scoring=scoring, cv=10, verbose=5)
        #    grid.fit(new_feature_set, Y)
        #    print "======================================================="
        #    print("The best parameters for SVM and %d are %s with a score of %0.6f"
        #          % (i, grid.best_params_, grid.best_score_))
        #
        #    if grid.best_score_ > best_score_total:
        #        print("SVM with %d features is best overall now." % (i))
        #        best_score_total = grid.best_score_
        #        best_params_total = grid.best_params_
        #        best_model_total = "SVM with " + str(i) + " features"
        #
        #    #RFR
        #    #rfr = RandomForestRegressor()
        #    #parameters = {'n_estimators': [120,300,500,800,1200], 'max_depth': [5,8,15,25,30,None], 'min_samples_split': [1,2,5,10,15,100], 'min_samples_leaf': [1,2,5,10], 'max_features':['log2','sqrt',None]}
        #    #grid = GridSearchCV(rfr, param_grid=parameters, scoring=scoring, cv=5, verbose=5)
        #    #grid.fit(X, Y)
        #    #print "==============================="
        #    #print "Best Hyperparameter Values:::"
        #    #print("The best parameters are %s with a score of %0.6f"
        #    #      % (grid.best_params_, grid.best_score_))
        #    #
        #    #if grid.best_score_ > best_score_total:
        #    #    best_score_total = grid.best_score_
        #    #    print "rfr"
        ##    best_model_total = "rfr"
        #
        #    print "Best model for BankQues User data ", best_model_total
        #    print "Best params ", best_params_total
        #    print "Best score ", best_score_total
        #
        #    filename_grid = DIR + 'best_gist.pkl'
        #    print 'nn_gist: ', 'saving file ...', filename_grid
        #    joblib.dump(best_params_total, filename_grid)
        #    pickle.dump([best_model_total, best_score_total], open(DIR+'best.p', 'w'))



    def load_clf(self):
        DIR = DIR = c.PATH_RESOURCES_HOME + 'results/'
        filename_grid = DIR + 'best_svm_gist_coarse.pkl'
        print 'svm_gist: ', 'found file ...', filename_grid
        grid = joblib.load(filename_grid)
        best_score_coarse = grid.best_score_
        best_param_coarse = grid.best_params_

        filename_fine = DIR + 'best_svm_gist_fine.pkl'
        print 'svm_gist: ', 'found file ...', filename_fine
        grid_fine = joblib.load(filename_fine)

        if grid_fine.best_score_ > best_score_coarse:
            print "\n Found a better fine tunes parameter set:"
            best_score = grid_fine.best_score_
            best_params = grid_fine.best_params_
        else:
            best_score = best_score_coarse
            best_params = best_param_coarse
        print("The final best parameters are %s with a score of %0.6f" % (best_params, best_score))

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

        #return n_features_train, labels_train, n_features_test, labels_test
        return n_features_train[:100], labels_train[:100], n_features_test[:100], labels_test[:100]

sg = svm_gist()
X, y, XT, yT = sg.load_data()
#print (X > 0)
sg.hyperparameter_optimization_svm_gist(X, y)
sg.load_clf()
sg.hyperparameter_optimization_with_feature_selection(X, y)