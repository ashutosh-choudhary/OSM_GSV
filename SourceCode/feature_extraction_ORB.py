from PIL import Image
import constants as c
import util
import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

TRAIN = "/training/"
TEST = "/test/"
VAL = "/val/"

class features_ORB:

    v_set = TRAIN

    def __init__(self, set_type = TRAIN):
        self.v_set = set_type

    def extract_ORB_features(self):
        """
        Method to extract and pickle color_histogram_features. The final dataset
        would have a size of D = 500, N = (Total number of pictures in train/test
        folders)
        Also, calls method to shuffle data before pickling
        :return: complete path/filename of pickled file
        """
        print "feature_ext_orb: " + self.v_set
        class_names = util.util().get_labels()
        data_set = []
        for class_name in class_names:
            print "\nfeature_ext_orb: " + "Extracting orb features for class " + class_name + "\n"
            DIR = c.PATH_DATA_HOME + class_name + self.v_set
            k = 0
            for filename in os.listdir(DIR):
                k += 1
                if filename.endswith(".jpg"):
                    im = Image.open(DIR + filename)
                    print "feature_ext_orb: " + filename + " " + str(im.size[0]) + "x" + str(im.size[1])
                    try:
                        im = cv2.imread(DIR + filename)
                        # Initiate STAR detector
                        orb = cv2.ORB(nfeatures = 10)
                        # find the keypoints with ORB
                        kp = orb.detect(im, None)
                        # compute the descriptors with ORB
                        kp, des = orb.compute(im, kp)

                        # draw only keypoints location,not size and orientation
                        #img2 = cv2.drawKeypoints(image, kp, color=(0, 255, 0), flags=0)
                        #plt.imshow(img2), plt.show()
                        print str(des.shape)
                        if not des.shape == (10, 32):
                            continue
                    except:
                        continue
                    else:
                        #Extracting x1, x2 ... x499, x500. Will add labels in next step
                        data_set_row = des.flatten().tolist()
                        #Appending y_label data
                        data_set_row.append(c.LABEL_MAP[class_name])
                        data_set.append(data_set_row)
                        # A plug for testing
                        #if k > 2:
                        #    break

        print "\nfeature_ext_orb: " + "Dataset created"
        print "feature_ext_orb: " + "N = " + str(len(data_set))

        print "\nfeature_ext_orb: " + "Sending for shuffle..."
        data_set = self.shuffle_data(data_set)
        print "feature_ext_orb: " + "Shuffle successful!"

        print "\nfeature_ext_orb: " + "Sending for pickle..."
        pickle_file = self.pickle_data(data_set)
        print "feature_ext_orb: " + "Pickle data successful!"
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
        DIR = c.PATH_DATA_ORB_PICKLE + self.v_set
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        filename = self.v_set.split("/")[1] + ".p"
        complete_file = DIR + filename
        pickle.dump(data_set, open(complete_file, "wb"))
        return complete_file