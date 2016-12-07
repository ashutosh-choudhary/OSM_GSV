from PIL import Image
import leargist
import constants as c
import util
import os
import numpy as np
import pickle

TRAIN = "/training/"
TEST = "/test/"
VAL = "/val/"

class features_GIST:

    v_set = TRAIN

    def __init__(self, set_type = TRAIN):
        self.v_set = set_type

    def extract_GIST_features(self):
        """
        Method to extract and pickle gist_features. The final dataset
        would have a size of D = 960, N = (Total number of pictures in train/test
        folders)
        Also, calls method to shuffle data before pickling
        :return: complete path/filename of pickled file
        """
        print "feature_ext_gist: " + self.v_set
        class_names = util.util().get_labels()
        data_set = []
        for class_name in class_names:
            print "\nfeature_ext_gist: " + "Extracting gist features for class " + class_name + "\n"
            DIR = c.PATH_DATA_HOME + class_name + self.v_set
            k = 0
            for filename in os.listdir(DIR):
                k += 1
                if filename.endswith(".jpg"):
                    im = Image.open(DIR + filename)
                    print "feature_ext_gist: " + filename + " " + str(im.size[0]) + "x" + str(im.size[1])

                    #Extracting x1, x2 ... x959, x960. Will add labels in next step
                    data_set_row = leargist.color_gist(im).tolist()
                    #Appending y_label data
                    data_set_row.append(c.LABEL_MAP[class_name])
                    data_set.append(data_set_row)

        print "\nfeature_ext_gist: " + "Dataset created"
        print "feature_ext_gist: " + "N = " + str(len(data_set))

        #print "\nfeature_ext_gist: " + "Before shuffle stats: "
        #first_column = [row[0] for row in data_set]
        #second_column = [row[1] for row in data_set]
        #third_column = [row[2] for row in data_set]
        #last_column = [row[len(data_set[0]) - 1] for row in data_set]
        #print first_column
        #print second_column
        #print third_column
        #print last_column
        #print "feature_ext_gist: " + str(type(first_column[0])) + " " + str(type(second_column[0])) + " " + str(type(third_column[0]))
        #print "feature_ext_gist: " + str(type(last_column[0]))

        print "\nfeature_ext_gist: " + "Sending for shuffle..."
        data_set = self.shuffle_data(data_set)
        print "feature_ext_gist: " + "Shuffle successful!"

        #print "\nfeature_ext_gist: " + "After shuffle stats: "
        #first_column = [row[0] for row in data_set]
        #second_column = [row[1] for row in data_set]
        #third_column = [row[2] for row in data_set]
        #last_column = [row[len(data_set[0]) - 1] for row in data_set]
        #print first_column
        #print second_column
        #print third_column
        #print last_column
        #print "feature_ext_gist: " + str(type(first_column[0])) + " " + str(type(second_column[0])) + " " + str(type(third_column[0]))
        #print "feature_ext_gist: " + str(type(last_column[0]))

        print "\nfeature_ext_gist: " + "Sending for pickle..."
        pickle_file = self.pickle_data(data_set)
        print "feature_ext_gist: " + "Pickle data successful!"
        return pickle_file

    def shuffle_data(self, data_set):
        """
        Method to randomly shuffle the entire dataset row-wise such that data is not ordered by class
        :param data_set: list :the data set in question
        :return: list: shuffled data set
        """
        np_data_set = np.array(data_set)
        np.random.shuffle(np_data_set)
        y_labels = np_data_set[:,-1].astype(int).tolist()
        new_np_data_set = np_data_set[:,:-1].tolist()
        for i in range(len(y_labels)):
            new_np_data_set[i].append(y_labels[i])
        return new_np_data_set

    def pickle_data(self, data_set):
        """
        Method to pickle the entire dataset such that data can be simply loaded everytime and not calculated
        for each method run
        :param data_set: list :the data set in question
        :return: string: the absolute path of the dataset generated
        """
        DIR = c.PATH_DATA_PICKLE + self.v_set
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        filename = self.v_set.split("/")[1] + ".p"
        complete_file = DIR + filename
        pickle.dump(data_set, open(complete_file, "wb"))
        return complete_file