import scipy.io
import constants as c
import urllib2
import urllib
import json
import collections as collec
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_OF_ELEMS_PLOT = 20

class extract_images:

    def __init__(self):
        self

    def main(self, classes_dict):
        """
        Extract relevant labels and plot a bar graph
        The following is observed from mat:
        mat                     - dictionary with three keys "SUN", "__version__", "__header__", "__global__"
        ..["SUN"]               - an ndarray which is just an enclosement of size 1
        ..[0]                   - an ndarray of size 900, needed to unwrap the previous enclosement
        ..[classNum]            - an ndarray of size 1 which has enclosement of type name and all 900 class urls
        ..[dataType]            - (or dt) an np.void struct type of size 3, 0 : labelName, 1 : annotations.xml, 2: images.jpg
        ..[dt=0][0]             - a wrapper ndarray of size 1 to enclose labelName
        ..[dt=0][0][0]          - a numpy.unicode_ type of string to denote labelName and its first letter folder_name
        ..[dt=1][0]             - a wrapper ndarray of size 1 to enclose xml urls
        ..[dt=1][0][xmlI]       - an ndarray to contain dynamic number(number of images) of ndarray for xml urls
        ..[dt=2][0]             - a wrapper ndarray of size 1 to enclose jpeg urls
        ..[dt=2][0][jpgI]       - an ndarray to contain dynamic number(number of images) of ndarray for jpg urls
        ..[dt=2][0][jpgI][0]    - a wrapper ndarray of size 1 to enclose jpg url
        ..[dt=2][0][jpgI][0][0] - final url

        Range:
        "SUN"                   - "__version__", "__header__", "__global__"
        0                       - none
        classNum                - 0 - 899
        dataType                - 0 - 2
        dt=0, 0                 - none
        dt=1, 0                 - none
        dt=1, 0, xmls           - dynamic/no of images
        dt=2, 0, jpg            - dynamic/no of images
        dt=2, 0, jpg, 0         - none
        dt=2, 0, jpg, 0 , 0     - final url

        File Info:
        1.0
        MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Mon Jan 17 22:26:14 2011
        :param classes_dict:
        :return:
        """

        #Extracting matlab file to access image URLs from SUN database
        print "extract_images: " + "Loading mat..."

        mat = scipy.io.loadmat('../Data/SUN_urls.mat')
        for key, value in mat.items():
            print key + " " + str(len(mat[key]))

        print "extract_images: " + "Number of classes: "
        print "extract_images: " + str(len(mat["SUN"][0]))
        num_available_img_classes = len(mat["SUN"][0])
        print "extract_images: " + "Number of images per class: "
        print "extract_images: " + str(len(mat["SUN"][0][0][2][0]))

        #print "extract_images: " + mat["SUN"][0][899][2][0][0][0]
        #print "extract_images: " + str(type(mat["SUN"][0][899][2][0][0][0]))
        #print "extract_images: " + str(len(mat["SUN"][0][899][2][0][0][0]))

        #Getting all available image labels from SUN database
        available_img_labels = []
        available_img_counts = []
        images_dict = {}

        for img in range(0, num_available_img_classes):
            available_img_labels.append(mat["SUN"][0][img][0][0])
            available_img_counts.append(len(mat["SUN"][0][img][2][0]))

        for img in range(0, num_available_img_classes):
            path_snippet = available_img_labels[img].split("\\")
            available_img_labels[img] = (path_snippet[1] + ' ' + path_snippet[2]).replace("_", " ") if len(path_snippet) == 3 else path_snippet[1].replace("_", " ")
            images_dict[available_img_labels[img]] = available_img_counts[img]

        #Getting a subset of image class labels which is a subset of image database corresponding to amenties tag
        corresponding_labels = self.find_corresponding_labels(classes_dict, available_img_labels, available_img_counts)


        # Sort both dictionaries and send for plot
        classes_sorted = sorted(classes_dict.items(), key=lambda x: x[1], reverse=True)
        images_dict_sorted = sorted(images_dict.items(), key=lambda x: x[1], reverse=True)
        corresponding_labels_sorted = sorted(corresponding_labels.items(), key=lambda x: x[1][1], reverse=True)
        od_classes = collec.OrderedDict(classes_sorted[:NUM_OF_ELEMS_PLOT])
        od_images = collec.OrderedDict(images_dict_sorted[:NUM_OF_ELEMS_PLOT])
        od_corresponding_labels = collec.OrderedDict(corresponding_labels_sorted[:NUM_OF_ELEMS_PLOT])

        self.plot_class_data_availability(od_classes)
        self.plot_image_data_availability(od_images)
        self.plot_class_with_image_data(classes_dict, images_dict)

        ############# Downloading the dataset ###############

        #write a function to do this
        selected_classes =  [u'p\\parking_lot', u'c\\church\\outdoor', u'g\\gas_station', u's\\shopfront']
        class_folders = [x.split("\\")[1] for x in selected_classes]
        for class_label in range(0, num_available_img_classes):
            if mat["SUN"][0][class_label][0][0] in selected_classes:
                num_images = len(mat["SUN"][0][class_label][2][0])
                for i in range(num_images):
                    url = mat["SUN"][0][class_label][2][0][i][0]
                    print "extract_images: downloading " + url
                    try:
                        response = urllib2.urlopen(url)
                    except urllib2.HTTPError, e:
                        print "extract_images: Error: Couldn't open url " + url + " due to error code " + str(e.code)
                    else:
                        rel_path = class_folders[selected_classes.index(mat["SUN"][0][class_label][0][0])]
                        local_path = "../Data/images/" + rel_path + "/"
                        file_name = rel_path + "_" + str(i) + ".jpg"
                        if not os.path.exists(local_path):
                            os.makedirs(local_path)
                        full_filename = os.path.join(local_path, file_name)
                        print "extract_images: Downloading image " + url
                        urllib.urlretrieve(url, full_filename)


    def plot_class_data_availability(self, dict_plot):
        """
        Method to plot top 20 wiki verified classes in OSM data with their number of tags
        :param dict_images: A dictionary containing the data described
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bin = np.arange(len(dict_plot))
        plt.bar(bin, dict_plot.values(), align='center')
        plt.xticks(bin, dict_plot.keys(), rotation='vertical')
        plt.xlim([-0.5,bin.size])
        plt.title("Count of class tag")
        plt.xlabel("Classes")
        plt.ylabel("Number of tags in OSM data")
        plt.show()

    def plot_image_data_availability(self, dict_images):
        """
        Method to plot images with their count values
        :param dict_plot: A dictionary of all classes and the number of avialble OSM tags
        :param dict_images: A didctionary of all image class and the number of available Image tags
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bin = np.arange(len(dict_images))
        plt.bar(bin, np.array(dict_images.values()), align='center')
        plt.xlim([-0.5,bin.size])
        plt.xticks(bin, dict_images.keys(), rotation='vertical')
        plt.title("Count of images")
        plt.xlabel("Image Labels")
        plt.ylabel("Number of images with class labels")
        plt.show()

    def plot_class_with_image_data(self, dict_plot, dict_images):
        """
        Method to plot split bar graphs with both number of OSM
        tags ond number of available images in log space
        :param dict_plot:
        :param dict_images:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bar_width = 0.45
        num_labels = []
        num_images = []
        labels = []

        for label1, label2 in c.TAG_EQUIVALENCE.items():
            labels += [label1]
            num_labels += [dict_plot[label1]]
            num_images += [dict_images[label2]]

        num_labels = np.array(num_labels)
        num_images = np.array(num_images)
        labels = np.array(labels)

        bin = np.arange(len(labels))
        rects1 = plt.bar(bin, np.log10(num_labels),bar_width, label='log(Num of Tags)', align='center', color='b')
        rects2 = plt.bar(bin + bar_width, np.log(num_images),bar_width, label='log(Num of Images)', align='center', color='r')
        plt.xlim([-bar_width,bin.size])
        plt.xticks(range(len(labels)), labels, rotation='vertical')
        plt.title("Trade off between available data and class labels")
        plt.xlabel("Class labels")
        plt.ylabel("Number of tags/ Number of available images")
        plt.legend()
        plt.show()

    def find_corresponding_labels(self, classes_dict, img_data, img_cnt):
        """
        Image manual label sync using a constant dictionary defined
        :param classes_dict: A dictionary of all classes and the number of avialble OSM tags
        :param img_data: A didctionary of all image class and the number of available Image tags
        :param img_cnt: Image counts
        :return:
        """
        class_word = classes_dict.keys()
        class_word_cnt = classes_dict.values()
        for j in range(0, len(class_word)):
            class_word[j] = class_word[j].replace("_", " ")
            #print "extract_images: c: " + class_word[j] + " " + str(class_word_cnt[j])

        for i in range(0, len(img_data)):
    #        img_data[i] = img_data[i].split("\\")[1].replace("_", " ")
            print "extract_images: i: " + img_data[i] + " " + str(img_cnt[i])

        synonyms_dict = {}
        synonyms_dict_list = {}

        for cw in range(0, len(class_word)):
            if not synonyms_dict.has_key(class_word[cw]):
                synonyms_dict[class_word[cw]] = [class_word[cw]]

        for labels, values in synonyms_dict.items():
            synonyms_dict_list[labels] = [[],0]
            #write a function here to find good intersections
            for class_check in values:
                if class_check in img_data:
                    index = img_data.index(class_check)
                    synonyms_dict_list[labels][0].append((img_data[index],img_cnt[index]))
                    synonyms_dict_list[labels][1] += img_cnt[index]

        return synonyms_dict_list

    def find_corresponding_labels_synonyms(self, classes_dict, img_data, img_cnt):
        """
        Image automatic label sync using thesaurus API
        :param classes_dict:
        :param img_data:
        :param img_cnt:
        :return:
        """
        class_word = classes_dict.keys()
        class_word_cnt = classes_dict.values()
        for j in range(0, len(class_word)):
            class_word[j] = class_word[j].replace("_", " ")
            #print "extract_images: c: " + class_word[j] + " " + str(class_word_cnt[j])

        for i in range(0, len(img_data)):
    #        img_data[i] = img_data[i].split("\\")[1].replace("_", " ")
            print "extract_images: i: " + img_data[i] + " " + str(img_cnt[i])

        synonyms_dict = {}
        synonyms_dict_list = {}
        # {'hut': [[('t', 10), ('s', 5), ('a', 7), ('m', 19)], 22]}
        for cw in range(0, len(class_word)):
            if not synonyms_dict.has_key(class_word[cw]):
                synonyms_dict[class_word[cw]] = [class_word[cw]]
            turl = c.THESAURUS_URL %class_word[cw]
            response = urllib.urlopen(turl)
            raw_data = json.loads(response.read())

            if not raw_data.has_key("error"):
                num_respose_fields = len(raw_data["response"])
                for rf in range(0, num_respose_fields):
                    if raw_data["response"][rf]["list"]["category"] == u'(noun)':
                        synonyms_dict[class_word[cw]] += raw_data["response"][rf]["list"]["synonyms"].split("|")
        #print synonyms_dict

        for labels, values in synonyms_dict.items():
            synonyms_dict_list[labels] = [[],0]

            #write a function here to find good intersections

            for class_check in values:
                if class_check in img_data:
                    index = img_data.index(class_check)
                    synonyms_dict_list[labels][0].append((img_data[index],img_cnt[index]))
                    synonyms_dict_list[labels][1] += img_cnt[index]

        return synonyms_dict_list