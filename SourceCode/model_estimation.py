import pickle
import constants as c
import numpy as np

PIPELINE_TYPE_TRAIN = "TRAIN"
PIPELINE_TYPE_TEST = "TEST"

class model_estimation:

    #v_set = PIPELINE_TYPE_TRAIN

    def __init__(self, mode = PIPELINE_TYPE_TRAIN):
       self

    def base_main(self):

        # Load the Data
        train_data = pickle.load(open(c.DATA_TRAINING_PICKLED, "rb"))
        test_data = pickle.load(open(c.DATA_TEST_PICKLED, "rb"))
        print "\n\n", "model_estimation: base_main: ", "Loaded pickled data successfully"
        features_train = np.array(train_data)[:,:-1]
        labels_train = np.array(train_data)[:,-1].astype(int)
        features_test = np.array(test_data)[:,:-1]
        labels_test = np.array(test_data)[:,-1].astype(int)
        print "model_estimation: base_main: ", ":", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape


