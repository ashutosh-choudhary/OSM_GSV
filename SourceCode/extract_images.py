import scipy.io
import constants as c
import urllib
import json
import collections as collec
import matplotlib.pyplot as plt
import numpy as np

NUM_OF_ELEMS_PLOT = 20

class extract_images:

    def __init__(self):
        self

    def main(self, classes_dict):

        print "extract_images: " + "Loading mat..."

        mat = scipy.io.loadmat('../Data/SUN_urls.mat')
        for key, value in mat.items():
            print key + " " + str(len(mat[key]))

        """
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
        """
        #url = mat["SUN"][0][0][2][0][0][0]
        #print url
        #urllib.urlretrieve(url, "../Data/images/test_image.jpg")

        print "extract_images: " + "Number of classes: "
        print "extract_images: " + str(len(mat["SUN"][0]))
        num_available_classes = len(mat["SUN"][0])
        print "extract_images: " + "Number of images per class: "
        print "extract_images: " + str(len(mat["SUN"][0][0][2][0]))

        print "extract_images: " + mat["SUN"][0][899][2][0][0][0]
        print "extract_images: " + str(type(mat["SUN"][0][899][2][0][0][0]))
        print "extract_images: " + str(len(mat["SUN"][0][899][2][0][0][0]))
        #http: // thesaurus.altervista.org / thesaurus / v1?key = unXYIXL2B0RpNA7V8XuG & word = fan & language = en_US & output = xml
        #turl = c.THESAURUS_URL+"?key="+c.THESAURUS_KEY+"&word="+class_word+"&language="+c.THESAURUS_LANG+"&output=json"
        classes = classes_dict.keys()
        print "extract_images: " + classes[0]

        available_img_labels = []
        available_img_counts = []
        print num_available_classes
        for img in range(0, num_available_classes):
            available_img_labels.append(mat["SUN"][0][img][0][0])
            available_img_counts.append(len(mat["SUN"][0][img][2][0]))
            #print mat["SUN"][0][img]
        corresponding_labels = self.find_corresponding_labels(classes, available_img_labels, available_img_counts)


        # Sort both dictiornaries and send for plot
        corresponding_labels_sorted = sorted(corresponding_labels.items(), key=lambda x: x[1][1], reverse=True)
        classes_sorted = sorted(classes_dict.items(), key=lambda x: x[1], reverse=True)
        od_corresponding_labels = collec.OrderedDict(corresponding_labels_sorted[:NUM_OF_ELEMS_PLOT])
        od_classes = collec.OrderedDict(classes_sorted[:NUM_OF_ELEMS_PLOT])
        self.plot_class_data_availability(od_classes)
        self.plot_image_data_availability(od_corresponding_labels)
        self.plot_class_with_image_data(classes_dict, collec.OrderedDict(corresponding_labels_sorted))

        return ['parking lot', 'church', 'fuel', 'shopfront', 'theater']

    def plot_class_data_availability(self, dict_plot):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bin = np.arange(len(dict_plot))
        plt.bar(bin, dict_plot.values(), align='center')
        plt.xticks(bin, dict_plot.keys(), rotation='vertical')
        plt.xlim([0,bin.size])
        plt.title("Count of class tag")
        plt.xlabel("Classes")
        plt.ylabel("Number of tags in OSM data")
        plt.show()

    def plot_image_data_availability(self, dict_images):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bin = np.arange(len(dict_images))
        plt.bar(bin, np.array(dict_images.values())[:,1], align='center')
        plt.xlim([0,bin.size])
        plt.xticks(bin, dict_images.keys(), rotation='vertical')
        plt.title("Count of images")
        plt.xlabel("Image Labels")
        plt.ylabel("Number of images with class labels")
        plt.show()

    def plot_class_with_image_data(self, dict_plot, dict_images):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        label_index = 0
        dict_tags = {}
        bar_width = 0.45

        for class_label, tag_count in dict_plot.items():
            class_label = class_label.replace("_"," ")
            image_index = dict_images.keys().index(class_label)
            total_index = label_index + image_index
            dict_tags[class_label] = [tag_count, dict_images[class_label][1], total_index]

        tags_sorted = sorted(dict_tags.items(), key=lambda  x: x[1][2])
        od_tags_labels = collec.OrderedDict(tags_sorted[:NUM_OF_ELEMS_PLOT])
        num_labels = np.array(od_tags_labels.values())[:, 0]
        num_images = np.array(od_tags_labels.values())[:, 1]
        bin = np.arange(len(od_tags_labels))
        rects1 = plt.bar(bin, num_labels*100/np.mean(num_labels),bar_width, label='Num of Tags', align='center', color='b')
        rects2 = plt.bar(bin + bar_width, num_images*100/np.mean(num_images),bar_width, label='Numn of Images', align='center', color='r')
        plt.xlim([0,bin.size])
        plt.xticks(range(len(od_tags_labels)), od_tags_labels.keys(), rotation='vertical')
        plt.title("Trade off between available data and class labels")
        plt.xlabel("Class labels")
        plt.ylabel("Number of tags/ Number of available images")
        plt.legend()
        plt.show()

    def find_corresponding_labels(self, classes, img_data, img_cnt):
        class_word = classes
        for j in range(0, len(classes)):
            classes[j] = classes[j].replace("_", " ")
            print "extract_images: c: " + classes[j]

        for i in range(0, len(img_data)):
            img_data[i] = img_data[i].split("\\")[1].replace("_", " ")
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

ei = extract_images()