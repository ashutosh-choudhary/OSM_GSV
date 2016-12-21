import os
import constants as c
import csv
import json

class util:

    def __init__(self):
        self

    def get_labels(self):
        """
        Method to get the labels from directory
        :return:
        """
        class_names = os.listdir(c.PATH_DATA_HOME)
        if class_names.__contains__('.DS_Store'):
            class_names.remove('.DS_Store')
        return class_names

    def get_class_dict(self):
        """
        Method to get the keys and values associated with each class
        :return:
        """
        class_names = self.get_labels()
        values = range(len(class_names))
        dict_labels = dict(zip(class_names, values))

        return dict_labels
