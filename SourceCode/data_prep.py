import os
import constants as c
import util
VAL_SPLIT_RATIO = .2
TEST_SPLIT_RATIO = .2

class data_prep:

    def __init__(self):
        self

    def separate_train_test_data(self):
        """
        Method to separate downloaded images into training and test sets.
        The ratio of split is as follows:
        Test:        20% of Total
        Training:    80% of Total
        :return:    none
        """

        class_names = util.util().get_labels()

        for class_name in class_names:
            DIR = c.PATH_DATA_HOME + class_name
            if os.listdir(DIR).__contains__('.DS_Store'):
                os.listdir(DIR).remove('.DS_Store')
            total_img_in_class = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            images_in_test = int(TEST_SPLIT_RATIO * total_img_in_class)

            print class_name + " " + str(total_img_in_class)
            print str(images_in_test)

            if not os.path.exists(c.PATH_DATA_HOME + class_name + "/test"):
                os.makedirs(c.PATH_DATA_HOME + class_name + "/test")
            if not os.path.exists(c.PATH_DATA_HOME + class_name + "/training"):
                os.makedirs(c.PATH_DATA_HOME + class_name + "/training")

            for img_num in range(total_img_in_class - 1):
                if 0 <= img_num < images_in_test:
                    os.rename(c.PATH_DATA_HOME + class_name + "/" + class_name + "_"+str(img_num)+".jpg", c.PATH_DATA_HOME + class_name + "/" + "test/"+ class_name + "_" + str(img_num)+".jpg")
                else:
                    os.rename(c.PATH_DATA_HOME + class_name + "/" + class_name + "_" + str(img_num) + ".jpg", c.PATH_DATA_HOME + class_name + "/" + "training/" + class_name + "_" + str(img_num) + ".jpg")

    def separate_train_val_test_data(self):
        """
        Method to separate downloaded images into training, validation and test sets.
        The ratio of split is as follows:
        Test:        20% of Total
        Validation:  16% of Total
        Training:    64% of Total
        :return:    none
        """
        class_names = os.listdir(c.PATH_DATA_HOME)

        if class_names.__contains__('.DS_Store'):
            class_names.remove('.DS_Store')
        print class_names

        for class_name in class_names:
            DIR = c.PATH_DATA_HOME + class_name
            total_img_in_class = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            images_in_test = int(TEST_SPLIT_RATIO * total_img_in_class)
            images_in_val = int(VAL_SPLIT_RATIO * (total_img_in_class - images_in_test))

            print class_name + " " + str(total_img_in_class)
            print str(images_in_test) + " " + str(images_in_val)

            if not os.path.exists(c.PATH_DATA_HOME + class_name + "/test"):
                os.makedirs(c.PATH_DATA_HOME + class_name + "/test")
            if not os.path.exists(c.PATH_DATA_HOME + class_name + "/val"):
                os.makedirs(c.PATH_DATA_HOME + class_name + "/val")
            if not os.path.exists(c.PATH_DATA_HOME + class_name + "/training"):
                os.makedirs(c.PATH_DATA_HOME + class_name + "/training")

            for img_num in range(total_img_in_class):
                if 0 <= img_num < images_in_test:
                    os.rename(c.PATH_DATA_HOME + class_name + "/" + class_name + "_"+str(img_num)+".jpg", c.PATH_DATA_HOME + class_name + "/" + "test/"+class_name + "_" + str(img_num)+".jpg")
                elif images_in_test <= img_num < images_in_test + images_in_val:
                    os.rename(c.PATH_DATA_HOME + class_name + "/" + class_name + "_" + str(img_num) + ".jpg", c.PATH_DATA_HOME + class_name + "/" + "val/" + class_name + "_" + str(img_num) + ".jpg")
                else:
                    os.rename(c.PATH_DATA_HOME + class_name + "/" + class_name + "_" + str(img_num) + ".jpg", c.PATH_DATA_HOME + class_name + "/" + "training/" + class_name + "_" + str(img_num) + ".jpg")

dp = data_prep()
dp.separate_train_test_data()

