import pickle
import constants as c
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import json
import math
import time
import os
from sklearn import preprocessing

PIPELINE_TYPE_TRAIN = "TRAIN"
PIPELINE_TYPE_TEST = "TEST"
seed = 7
validation_size = 0.20
num_folds = 10
scoring = 'accuracy'

models = [('LR', LogisticRegression()),
          ('LDA', LinearDiscriminantAnalysis()),
          ('QDA', QuadraticDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('DTC', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('LSVM', SVC(kernel="linear", verbose=True, C=0.025)),
          ('RSVM', SVC()),
          ('RF', RandomForestClassifier()),
          ('NN', MLPClassifier(alpha=1)),
          ('AB', AdaBoostClassifier())]

#best_set_classifiers = {'LR', 'LDA', 'QDA', 'KNN', 'DTC', 'NB', 'LSVM', 'RSVM', 'RF', 'NN', 'AB'}
best_set_classifiers = {'LR', 'LDA', 'QDA', 'KNN', 'DTC', 'NB', 'RSVM', 'RF', 'NN', 'AB'}

class evaluate_classifiers:

    #v_set = PIPELINE_TYPE_TRAIN
    FEATURE_TYPE = 'GIST'
    NORM = False

    def __init__(self, mode = PIPELINE_TYPE_TRAIN, feature_type=c.FEATURE_TYPE_LIST[0], normalization=False):
       self.FEATURE_TYPE = feature_type
       self.NORM = normalization

    def main(self):
        # Load the Data
        features_train, labels_train, features_test, labels_test = self.load_data(self.FEATURE_TYPE)

        print "evaluate_classifiers: ", "main:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape

        # Describe Data
        print "\n-------------- Describe Data -------------------------\n"
        for k, v in self.describe_data(features_train).items(): print k, ": ", v
        print "\n Labels count: \n", self.describe_classes(labels_train)
        print "\n Data sample: features \n", self.head(features_train, 20)
        print "\n Data sample: labels \n", self.head(labels_train, 20)

        # Evaluate best model with default hyperparameters
     #   print "\n-------------- Evaluate Default Models -------------------------\n"
     #   i = 0
     #   best_set_classifiers = {'KNN', 'SVM', 'LR', 'LDA', 'DTC', 'NB'}
     #   comparision = self.evaluate_all_models(features_train, labels_train)
     #   for key, value in sorted(comparision.items(), key=operator.itemgetter(1), reverse=True):
     #       print key, ": ", value
     #
     #   # Examine Accuracy of the three chosen classifiers
     #   print "\n-------------- Three Classifiers (Accuracy) -------------------------\n"

        comparision_best = self.default_hparam_accuracy(features_train, labels_train, best_set_classifiers)
        for key, value in sorted(comparision_best.items(), key=operator.itemgetter(1), reverse=True):
            print key, ": (Acc Mean, Acc Std) ", value

        # Examine Training and Prediction times of the three chosen classifiers
        print "\n-------------- Three Classifiers (Training Time) -------------------------\n"

        comparision_best_time = self.default_hparam_times(features_train, labels_train, features_test,
                                                        best_set_classifiers)
        for key, value in sorted(comparision_best_time.items(), key=operator.itemgetter(1), reverse=False):
            print key, ": (TrainTime, PredictionTime) ", value

    def get_sub_list(self, list_main, list_sub_set):
        """
        Generate a sub list from a the master list provided. Used for
        chosing three out of six classifiers evaluated
        :param list_main: The master set
        :param list_sub_set: The subset labels
        :return: subset dictionary
        """
        tmp_dict = dict(list_main)
        tmp_dict_sub = {name:func for name,func in tmp_dict.items() if name in list_sub_set}
        return tmp_dict_sub.items()

    def load_data(self, feature_type):

        # Load the Data
        if feature_type == c.FEATURE_TYPE_LIST[0]:
            filename_train = c.DATA_TRAINING_GIST_PICKLED
            filename_test = c.DATA_TEST_GIST_PICKLED
        elif feature_type == c.FEATURE_TYPE_LIST[1]:
            filename_train = c.DATA_TRAINING_COH_PICKLED
            filename_test = c.DATA_TEST_COH_PICKLED
        else:
            filename_train = c.DATA_TRAINING_ORB_PICKLED
            filename_test = c.DATA_TEST_ORB_PICKLED

        train_data = pickle.load(open(filename_train, "rb"))
        test_data = pickle.load(open(filename_test, "rb"))
        print "\n\n", "model_estimation: base_main: ", "Loaded pickled data successfully"
        features_train = np.array(train_data)[:,:-1]
        labels_train = np.array(train_data)[:,-1].astype(int)
        features_test = np.array(test_data)[:,:-1]
        labels_test = np.array(test_data)[:,-1].astype(int)
        #print "model_estimation: base_main: ", ":", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape
        if self.NORM:
            n_features_train = preprocessing.scale(features_train)
            n_features_test = preprocessing.scale(features_test)
        else:
            n_features_train = features_train
            n_features_test = features_test

        return n_features_train, labels_train, n_features_test, labels_test


    def evaluate_all_models(self, X, Y):
        """
        Evaluating a master set of classifiers to list their accuracies for a given dataset
        :param X: The feature vector
        :param Y: The class vector
        :return: Dictionary of class labels and their mean accuracies and standard deviations
        """
        results = []
        names = []
        num_instances = len(X)
        dict_model = {}
        for name, model in models:
            kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
            cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            dict_model[name] = [cv_results.mean() * 100, cv_results.std() * 2]
        return dict_model


    def plot_data_characteristics(self, dataset):
        """
        Plot characteristics of data such as Count, Mean, Std ect
        :param dataset: The dataset to be analyzed and plotted
        :return:
        """
        if self.NORM:
            fileExt = '_norm'
            label_chng = ' with normalization'
        else:
            fileExt = ''
            label_chng = ' without normalization'

        number_of_elements = dataset['Avg'].shape[0]
        # Construct List of feature labels
        list_labels = []
        for i in range(1, number_of_elements + 1, 1):
            list_labels.append('X' + str(i))
        # Plot Mean of features
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_loc = np.arange(number_of_elements)
        width = .35 * (16 / number_of_elements)
        plt.bar(x_loc, dataset['Avg'], width=width, yerr=dataset['Std'])
        plt.ylabel('Mean of Features')
        plt.xlabel('Features')
        plt.xticks([])
        plt.tick_params(axis='x')
        plt.title(self.FEATURE_TYPE + ': Description of Features: Mean and Std' + label_chng)
        plt.margins(0.02)
        low = min(dataset['Avg'])
        high = max(dataset['Avg'])
        plt.ylim([math.ceil(low - 0.5 * (high - low)), math.ceil(high + 0.5 * (high - low))])
        print "\n Plotting Data Characteristics ..."
        plt.savefig(c.PATH_FIGURES + self.FEATURE_TYPE + fileExt + '_features_Mean_Std.png')
        print c.PATH_FIGURES + self.FEATURE_TYPE + fileExt + '_features_Mean_Std.png saved.\n'
        plt.gcf().clear()


    def plot_test_accuracy(self, model_dict):
        """
        Plot the accuracy or cross-validation score of dataset for all models in model_dict
        :param model_dict: The dictionary of classifiers
        :return:
        """
        if self.NORM:
            fileExt = '_norm'
            label_chng = ' with normalization'
        else:
            fileExt = ''
            label_chng = ' without normalization'

        filename = self.save_data(model_dict, "accuracy")
        print "Data saved in file: ", filename
        x_loc = np.arange(len(model_dict))
        width = .35
        list_label = []
        mean = []
        std = []
        for name, value in model_dict.items():
            list_label.append(name)
            mean.append(value[0])
            std.append(value[1])
        plt.bar(x_loc, mean, width=width, yerr=std)
        plt.ylabel('Accuracy')
        plt.xlabel('Classifier Type')
        plt.xticks(x_loc + width / 2, list_label)
        plt.tick_params(axis='x')
        plt.title(self.FEATURE_TYPE + ': Test Accuracy of Classifiers' + label_chng)
        plt.margins(0.02)
        low = min(mean)
        high = max(mean)
        plt.ylim([math.ceil(low - 0.5 * (high - low)), math.ceil(high + 0.5 * (high - low))])
        print "\n Plotting Accuracy ..."
        plt.savefig('../Figures/' + self.FEATURE_TYPE + fileExt + '_Accuracy_Default.png')
        print "../Figures/" + self.FEATURE_TYPE + fileExt + "_Accuracy_Default.png saved.\n"
        plt.gcf().clear()

    def default_hparam_accuracy(self, X, Y, model_sub):
        """
        Comparision between classifiers in model_sub subset in terms of
        their accuracy with default hyperparameters used for classifiers
        :param X: The feature vector
        :param Y: The class labels
        :param model_sub: Subset of classifiers to be compared
        :return: Dictionary of class labels and their mean accuracies
        """
        model_subset = self.get_sub_list(models, model_sub)
        results = []
        names = []
        num_instances = len(X)
        dict_model = {}
        for name, model in model_subset:
            print "\n", "evaluate_classifiers: ", "accuracy: ", "Now running... ", name
            kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
            cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            dict_model[name] = [cv_results.mean()*100, cv_results.std()*2]
            print "evaluate_classifiers: ", "accuracy: ", "Completed ", name
        self.plot_test_accuracy(dict_model)
        return dict_model

    def default_hparam_times(self, x_train, y_train, x_test, model_sub):
        """
        Comparision between classifiers in model_sub subset in terms of
        their training and prediction times with default hyperparameters used for classifiers
        :param X: The feature vector
        :param Y: The class labels
        :param model_sub: Subset of classifiers to be compared
        :return: Dictionary of class labels and their training and prediction times
        """
        model_subset = self.get_sub_list(models, model_sub)
        names = []
        dict_model = {}
        for name, model in model_subset:
            print "\n", "evaluate_classifiers: ", "time: ", "Now running... ", name
            train_time_start = time.time()
            model.fit(x_train, y_train)
            train_time_stop = time.time()
            train_time = train_time_stop - train_time_start
            #print "\n", name, ": Training time: ", train_time
            predict_time_start = time.time()
            predictions = model.predict(x_test)
            predict_time_stop = time.time()
            predict_time = predict_time_stop - predict_time_start
            #print "\n", name, ": Prediction time: ", predict_time
            names.append(name)
            dict_model[name] = [train_time,predict_time]
            print "evaluate_classifiers: ", "time:", "Completed ", name
        self.plot_training_time(dict_model)
        self.plot_prediction_time(dict_model)
        return dict_model

    def plot_training_time(self, model_dict):
        """
        Plot the training time of all models in model_dict
        :param model_dict: The dictionary of classifiers
        :return:
        """
        if self.NORM:
            fileExt = '_norm'
            label_chng = ' with normalization'
        else:
            fileExt = ''
            label_chng = ' without normalization'

        filename = self.save_data(model_dict, "traintime")
        print "Data saved in file: ", filename
        x_loc = np.arange(len(model_dict))
        width = .35
        list_label = []
        train_time = []
        for name, value in model_dict.items():
            list_label.append(name)
            train_time.append(value[0])
        plt.bar(x_loc, train_time, width=width)
        plt.ylabel('Training Time')
        plt.xlabel('Classifier Type')
        plt.xticks(x_loc + width/2, list_label)
        plt.tick_params(axis='x')
        plt.title(self.FEATURE_TYPE + ': Training time of Classifiers'+label_chng)
        plt.margins(0.02)
        #low = min(train_time)
        #high = max(train_time)
        #plt.ylim([math.ceil(low - 0.5 * (high - low)), math.ceil(high + 0.5 * (high - low))])
        print "\n Plotting Training time ..."
        plt.savefig(c.PATH_FIGURES + self.FEATURE_TYPE + fileExt + '_Training_Time.png')
        print c.PATH_FIGURES + self.FEATURE_TYPE + fileExt + "_Training_Time.png saved.\n"
        plt.gcf().clear()

    def plot_prediction_time(self, model_dict):
        """
        Plot the prediction time of all models in model_dict
        :param model_dict: The dictionary of classifiers
        :param ques_no: The quesno in HW01. Used for title of graphs etc.
        :return:
        """
        if self.NORM:
            fileExt = '_norm'
            label_chng = ' with normalization'
        else:
            fileExt = ''
            label_chng = ' without normalization'

        filename = self.save_data(model_dict, "predtime")
        print "Data saved in file: ", filename
        x_loc = np.arange(len(model_dict))
        width = .35
        list_label = []
        predict_time = []
        for name, value in model_dict.items():
            list_label.append(name)
            predict_time.append(value[1])
        plt.bar(x_loc, predict_time, width=width)
        plt.ylabel('Prediction Time')
        plt.xlabel('Classifier Type')
        plt.xticks(x_loc + width/2, list_label)
        plt.tick_params(axis='x')
        plt.title(self.FEATURE_TYPE + ': Prediction time of Classifiers' + label_chng)
        plt.margins(0.02)
        #low = min(train_time)
        #high = max(train_time)
        #plt.ylim([math.ceil(low - 0.5 * (high - low)), math.ceil(high + 0.5 * (high - low))])
        print "\n Plotting Prediction time ..."
        plt.savefig(c.PATH_FIGURES + self.FEATURE_TYPE + fileExt + '_Prediction_Time.png')
        print c.PATH_FIGURES + self.FEATURE_TYPE + fileExt + "_Prediction_Time.png saved.\n"
        plt.gcf().clear()

    def head(self, data, num):
        """
        Get first num rows of data
        :param data: Dataset
        :param num: Number of rows required
        :return: List of first num rows
        """
        if type(data) == list and len(data) > 1:
            return [row[0:num] for row in data]
        elif type(data) == list and len(data) == 1:
            return data[:num]
        elif type(data) == np.ndarray and len(data.shape) > 1:
            return data[np.r_[:num],:]
        elif type(data) == np.ndarray and len(data.shape) == 1:
            return data[:num]

    def tail(self, data, num):
        if type(data) == list and len(data) > 1:
            return [row[num:] for row in data]
        elif type(data) == list and len(data) == 1:
            return data[num:]
        elif type(data) == np.ndarray and len(data) > 1:
            return data[:,num:]
        elif type(data) == np.ndarray and len(data) == 1:
            return data[num:]


    def describe_data(self, features):
        description = {}
        description['Cnt'] = features.shape[0]
        description['Avg'] = features.mean(axis=0)
        description['Std'] = features.std(axis=0)
        description['Min'] = features.min(axis=0)
        description['25%'] = np.percentile(features,25,axis=0)
        description['50%'] = np.percentile(features,50,axis=0)
        description['75%'] = np.percentile(features,75,axis=0)
        description['Max'] = features.max(axis=0)
        self.plot_data_characteristics(description)
        return description

    def describe_classes(self, labels):
        label_dict = {}
        for i in range(0, np.alen(labels), 1):
            if labels[i] in label_dict:
                label_dict[labels[i]] += 1
            else:
                label_dict[labels[i]] = 1
        return label_dict


    def save_data(self, save_dict, res_type):
        """
        Method to save the results such that they can be plotted later
        :param save_dict: list :the data set in question
        :return: string: the absolute path of the dataset generated
        """
        if self.NORM:
            fileExt = '_norm'
            label_chng = ' with normalization'
        else:
            fileExt = ''
            label_chng = ' without normalization'

        filename = res_type
        if self.FEATURE_TYPE == c.FEATURE_TYPE_LIST[0]:
            DIR = c.PATH_RES_GIST_TRAIN
            filename += fileExt + "_results_gist.json"
        elif self.FEATURE_TYPE == c.FEATURE_TYPE_LIST[1]:
            DIR = c.PATH_RES_COH_TRAIN
            filename += fileExt + "_results_coh.json"
        else:
            DIR = c.PATH_RES_ORB_TRAIN
            filename += fileExt + "_results_orb.json"
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        complete_file = DIR + filename
        json.dump(save_dict, open(complete_file, "w"))
        return complete_file