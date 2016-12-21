from PIL import Image
import leargist
import constants as c
import util
import os
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import gist_classifier_exp as gce
import util as u
import csv
import json
import warnings



class evaluate_model_on_gsv:

    def __init__(self):
        self

    def extract_gist_for_images_and_save(self):
        data_set =[]
        for subfolder in os.listdir(c.PATH_IMAGES_GSV):
            if subfolder == '.DS_Store':
                continue
            for file in os.listdir(c.PATH_IMAGES_GSV+subfolder):
                if file == '.DS_Store':
                    continue
                complete_filename = c.PATH_IMAGES_GSV + subfolder + "/" + file
                print "extracting gist for: ", complete_filename
                im = Image.open(complete_filename)
                gist_feature_row = leargist.color_gist(im).tolist()
                lat, lon = subfolder.split('x')
                file_mod = file.split('.')[0]
                label, heading = file_mod.rsplit('_', 1)
                data_row = [lat, lon, heading, label, gist_feature_row]
                print data_row
                data_set.append(data_row)
        self.pickle_data(c.FILENAME_AMHERST_GSV, data_set)
        print "All the data pickled succesfully! "

    def extract_gist_for_one_image(self, subfolder, filename):

        im = Image.open(c.PATH_IMAGES_GSV + subfolder + filename)
        print "feature_ext_gist: " + filename + " " + str(im.size[0]) + "x" + str(im.size[1])

        # Extracting x1, x2 ... x959, x960.
        data_set_row = leargist.color_gist(im).tolist()
        return

    def evaluate(self):
        data_set = self.read_gsv_gist_data()

        # Get a trained model for best parameters

        sg = gce.svm_gist()
        X, y, XT, yT = sg.load_data()
        svc = SVC(kernel='linear', C=0.01, gamma=0.01, probability=True, verbose=5)
        svc.fit(X, y)
        #print svc.classes_
        #print data_set[2]
        true_headings_dict = self.read_json_gsv()
        best_heading = 0
        correct, incorrect = 0, 0
        for i in range(len(data_set)):
            if (i + 1) % 6 == 0 or i == 0:
                if not i == 0:
                    print data_set[i-1][0], 'x', data_set[i-1][1], ' ' , best_heading, ' ', true_headings_dict[data_set[i-1][0]+'x'+data_set[i-1][1]]
                    if best_heading == true_headings_dict[data_set[i-1][0]+'x'+data_set[i-1][1]]:
                        print data_set[i-1]
                        correct += 1
                    else:
                        incorrect +=1
                best_probab = 0
                label_index = c.LABEL_MAP[data_set[i][3]]
               # print label_index
            features_test = data_set[i][4]
            n_features_test = preprocessing.scale(features_test)
            extracted_label = svc.predict([n_features_test])
            probabs = svc.predict_proba([n_features_test])
            #print probabs[0]
            #print probabs[0][label_index]
            if probabs[0][label_index] > best_probab:
                best_probab = probabs[0][label_index]
                best_heading = data_set[i][2]
        print 'correct % = ', (float(correct) / (correct + incorrect)) *100

    def get_best_model_trained(self):
        DIR = c.PATH_MODEL_HOME
        filename_coarse = DIR + c.FILENAME_BEST_SVM_COARSE
        print 'svm_gist: ', 'extracting file ...', filename_coarse
        clf_svm_coarse = joblib.load(filename_coarse)
        print clf_svm_coarse.best_params_
     #   print clf_svm_coarse.best_score_
     #   print clf_svm_coarse.best_estimator_

        filename_fine = DIR + c.FILENAME_BEST_SVM_FINE
        print 'svm_gist: ', 'extracting file ...', filename_fine
        clf_svm_fine = joblib.load(filename_fine)
     #   print clf_svm_fine.best_params_
     #   print clf_svm_fine.best_score_
     #   print clf_svm_fine.best_estimator_

        filename_nn_coarse = DIR + c.FILENAME_BEST_NN_COARSE
        print 'svm_gist: ', 'extracting file ...', filename_nn_coarse
        clf_nn_coarse = joblib.load(filename_nn_coarse)
     #   print clf_nn_coarse.best_params_
     #   print clf_nn_coarse.best_score_
     #   print clf_nn_coarse.best_estimator_
        return clf_svm_coarse.best_estimator_

    def read_gsv_gist_data(self):
        DIR = c.PATH_DATA_GSV
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        complete_file = DIR + c.FILENAME_AMHERST_GSV
        data_set = pickle.load(open(complete_file, 'rb'))
        return data_set

    def pickle_data(self, filename,  data_set):
        """
        Method to pickle the entire dataset such that data can be simply loaded everytime and not calculated
        for each method run
        :param data_set: list :the data set in question
        :return: string: the absolute path of the dataset generated
        """
        DIR = c.PATH_DATA_GSV
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        complete_file = DIR + filename
        pickle.dump(data_set, open(complete_file, "wb"))
        return complete_file


    def create_lat_long_csv(self):
        lat_long = []
        for subfolder in os.listdir(c.PATH_IMAGES_GSV):
            if subfolder == '.DS_Store':
                continue
            else:
                lat_long.append([subfolder, subfolder.split('x')[0], subfolder.split('x')[1]])
        if not os.path.exists(c.PATH_DATA_GSV):
            os.makedirs(c.PATH_DATA_GSV)
        filename = c.PATH_DATA_GSV + c.FILENAME_AMHERST_HEADING_VALS_GSV
        print 'create_lat_long_csv: ', 'Saved file: ', filename, '\n'
        with open(filename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(iter(lat_long))


    def read_lat_long_csv(self):
        data = []
        comlete_file_name = c.PATH_DATA_GSV + c.FILENAME_AMHERST_HEADING_VALS_GSV
        if os.path.exists(comlete_file_name):
            print 'read_lat_long_csv: ', 'Extracting file: ', comlete_file_name, '\n'
            with open(comlete_file_name, 'rb') as f:
                reader = csv.reader(f)
                try:
                    for row in reader:
                        data.append(row)
                except csv.Error as e:
                    print 'Couldn\'t read data for file %s line %d' % (comlete_file_name, reader.line_num)
            return data
        else:
            print 'read_lat_long_csv: No data in the directory ', comlete_file_name
        return []


    def create_json_from_csv(self):
        data = self.read_lat_long_csv()
        data_dict = {}
        for row in data:
            data_dict[row[0]] = row[3]
        newfilename = c.FILENAME_AMHERST_HEADING_VALS_GSV.split('.')[0] + '.json'
        complete_filename = c.PATH_DATA_GSV + newfilename
        json.dump(data_dict, open(complete_filename, 'w'))


    def read_json_gsv(self):
        newfilename = c.FILENAME_AMHERST_HEADING_VALS_GSV.split('.')[0] + '.json'
        complete_filename = c.PATH_DATA_GSV + newfilename
        return json.load(open(complete_filename, 'r'))

