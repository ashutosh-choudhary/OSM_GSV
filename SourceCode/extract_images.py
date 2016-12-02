import scipy.io
import constants as c
import urllib
import json

class extract_images:

    def __init__(self):
        self

    def main(self, classes):

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
        print "extract_images: " + classes[0]
        available_img_labels = []
        available_img_counts = []
        print num_available_classes
        for img in range(0, num_available_classes):
            available_img_labels.append(mat["SUN"][0][img][0][0])
            available_img_counts.append(len(mat["SUN"][0][img][2][0]))
            #print mat["SUN"][0][img]
        self.find_corresponding_labels(classes, available_img_labels, available_img_counts)


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
        print synonyms_dict

        for labels, values in synonyms_dict.items():
            synonyms_dict_list[labels] = [[],0]
            for class_check in values:
                if class_check in img_data:
                    index = img_data.index(class_check)
                    synonyms_dict_list[labels][0].append((img_data[index],img_cnt[index]))
                    synonyms_dict_list[labels][1] += img_cnt[index]

        synonyms_dict_list_sorted = sorted(synonyms_dict_list.items(), key=lambda x: x[1][1], reverse=True)
        print synonyms_dict_list_sorted



ei = extract_images()