# Author: Ashutosh Choudhary
# email: ashutoshchou@umass.edu
# Dec 2016

import extract_all_classes as eac
import extract_images as ei
import feature_extraction_GIST as gist
import feature_extraction_COH as coh
import feature_extraction_ORB as orb
import Evaluate_Classifiers as ec
import time
import constants as c
import plotting as ping
import extract_osm_data as eod
import gist_classifier_exp as gce
import coh_classifier_exp as cce
import extract_street_view_images as gsv_images
import evaluate_model_on_gsv as eval

class run_main(object):
    """
    The driver class for all the data scrapping and experimentation in
    other class file. Each part of the project can be run independently
    by providing the described parameters. However, to run an end-to-end
    pipeline you need to have the following data in the following path:
    <Insert path to a blog>
    """

    def __init__(self):
        self

    def main(self):
        """
        The main (driver) function which is called at the end of this executable file
        It has different sections of the project being called in succession.
        Please consider that it may take a huge amount of time for the entire pipeline to
        execute. In the order of ~14 hours. This involves all the image scrapping and
        model training
        :return: none
        """

    #    #All class labels processing
    #    classes = eac.extract_all_classes().main()     #These are top wiki verified classes in http://taginfo.openstreetmap.org/
    #
    #    #Extract images of top
    #    ei.extract_images().main(classes)
    #    self.extract_and_pickle_gist()
    #    self.extract_and_pickle_coh()
    #    self.extract_and_pickle_orb()
    #
    #    #Evaluate all classifiers
    #    self.eval_all_classifiers()
    #
    #    #Plot all combined plot
    #    self.plot_all_combined()
    #
    #    #Train and tune SVM and Neural Net with GIST feature set
    #    self.experiment_with_gist_features()
    #
    #    #Get the saved value of SVM and Neural Net
    #    svm, nn = self.load_best_models_with_gist_features()
    #
    #    #Plot the GIST and NN hyperparameter tuning
    #    #self.plot_gist_svm_tuning(svm)
    #    self.plot_gist_nn_tuning(nn)
    #
    #    #Train and tune SVM and Neural Net with GIST feature set
    #    self.experiment_with_coh_features()
    #
    #    sg = gce.svm_gist()
    #    sg.final_test_svm()
    #    sg.final_test_nn()
        coh = cce.svm_coh()
        coh.final_test_rf()
        coh.final_test_ab()
    #
    #    #Get the saved value of SVM and Neural Net
    #    rf, ab = self.load_best_models_with_coh_features()
    #
    #    #Plot the RF and Ab hyperparameter tuning
    #    self.plot_coh_rf_tuning(rf)
    #    self.plot_coh_ab_tuning(ab)
    #
    #    #Extract osm data
    #    self.extract_osm_data()
    #
    #    #Extract Street View Images using data collected from OSM data
    #    self.extract_gsv_images()

    #    self.eval_final()


    def extract_and_pickle_gist(self):
        # Pickle data
        start_pickling_time = time.time

        pickled_file = gist.features_GIST(gist.TRAIN).extract_GIST_features()
        print "The pickled training file is: " + pickled_file
        pickled_file = gist.features_GIST(gist.TEST).extract_GIST_features()
        print "The pickled training file is: " + pickled_file

        stop_pickling_time = time.time()

        time_pickling_time_s = stop_pickling_time - start_pickling_time
        time_pickling_time_m, time_pickling_time_s = divmod(time_pickling_time_s, 60)
        time_pickling_time_h, time_pickling_time_m = divmod(time_pickling_time_m, 60)
        print "pickle: " + "Time taken to process all classes: %d hours %02d mins %02d secs" % (time_pickling_time_h, time_pickling_time_m, time_pickling_time_s)


    def extract_and_pickle_coh(self):
        # Pickle data
        start_pickling_time = time.time()

        pickled_file = coh.features_COH(coh.TRAIN).extract_COH_features()
        print "The pickled training file is: " + pickled_file
        pickled_file = coh.features_COH(coh.TEST).extract_COH_features()
        print "The pickled training file is: " + pickled_file

        stop_pickling_time = time.time()

        time_pickling_time_s = stop_pickling_time - start_pickling_time
        time_pickling_time_m, time_pickling_time_s = divmod(time_pickling_time_s, 60)
        time_pickling_time_h, time_pickling_time_m = divmod(time_pickling_time_m, 60)
        print "pickle: " + "Time taken to process all classes: %d hours %02d mins %02d secs" % (time_pickling_time_h, time_pickling_time_m, time_pickling_time_s)


    def extract_and_pickle_orb(self):
        # Pickle data
        start_pickling_time = time.time()

        pickled_file = orb.features_ORB(orb.TRAIN).extract_ORB_features()
        print "The pickled training file is: " + pickled_file
        pickled_file = orb.features_ORB(orb.TEST).extract_ORB_features()
        print "The pickled training file is: " + pickled_file

        stop_pickling_time = time.time()

        time_pickling_time_s = stop_pickling_time - start_pickling_time
        time_pickling_time_m, time_pickling_time_s = divmod(time_pickling_time_s, 60)
        time_pickling_time_h, time_pickling_time_m = divmod(time_pickling_time_m, 60)
        print "pickle: " + "Time taken to process all classes: %d hours %02d mins %02d secs" % (time_pickling_time_h, time_pickling_time_m, time_pickling_time_s)

    def eval_all_classifiers(self):
        #Without data normalization
        # GIST with default hyperparameter models
        eval_class = ec.evaluate_classifiers(feature_type=c.FEATURE_TYPE_LIST[0])
        eval_class.main()
        # COH with default hyperparameter models
        eval_class = ec.evaluate_classifiers(feature_type=c.FEATURE_TYPE_LIST[1])
        eval_class.main()
        # ORB with default hyperparameter models
        eval_class = ec.evaluate_classifiers(feature_type=c.FEATURE_TYPE_LIST[2])
        eval_class.main()

        # With data normalization
        # GIST with default hyperparameter models
        eval_class = ec.evaluate_classifiers(feature_type=c.FEATURE_TYPE_LIST[0], normalization=True)
        eval_class.main()
        # COH with default hyperparameter models
        eval_class = ec.evaluate_classifiers(feature_type=c.FEATURE_TYPE_LIST[1], normalization=True)
        eval_class.main()
        # ORB with default hyperparameter models
        eval_class = ec.evaluate_classifiers(feature_type=c.FEATURE_TYPE_LIST[2], normalization=True)
        eval_class.main()

    def plot_all_combined(self):
        ping.plot_combined("accuracy")
        ping.plot_combined("traintime")
        ping.plot_combined("predtime")
        ping.plot_combined("accuracy", normalization=True)
        ping.plot_combined("traintime", normalization=True)
        ping.plot_combined("predtime", normalization=True)

    def extract_osm_data(self):
        ed = eod.extract_osm_data()
        # Save OSM data
        start_saving_time = time.time()

        ed.read_and_parse_osm()

        stop_saving_time = time.time()

        time_saving_time_s = stop_saving_time - start_saving_time
        time_saving_time_m, time_saving_time_s = divmod(time_saving_time_s, 60)
        time_saving_time_h, time_saving_time_m = divmod(time_saving_time_m, 60)
        print "save csv: " + "Time taken to process all classes: %d hours %02d mins %02d secs" % (time_saving_time_h, time_saving_time_m, time_saving_time_s)

    def extract_gsv_images(self):
        gsv = gsv_images.extract_street_view_images()
        # Save GSV images
        start_saving_time = time.time()

        gsv.extract_and_save_images()

        stop_saving_time = time.time()

        time_saving_time_s = stop_saving_time - start_saving_time
        time_saving_time_m, time_saving_time_s = divmod(time_saving_time_s, 60)
        time_saving_time_h, time_saving_time_m = divmod(time_saving_time_m, 60)
        print "save csv: " + "Time taken to save all GSV images: %d hours %02d mins %02d secs" % (time_saving_time_h, time_saving_time_m, time_saving_time_s)


    def experiment_with_gist_features(self):
         sg = gce.svm_gist()
         X, y, XT, yT = sg.load_data()
         sg.hyperparameter_optimization_svm_gist(X, y)
         sg.hyperparameter_optimization_mlcp_gist(X, y)

    def experiment_with_coh_features(self):
         coh = cce.svm_coh()
         X, y, XT, yT = coh.load_data()
         coh.hyperparameter_optimization_rf_coh(X, y)
         coh.hyperparameter_optimization_ab_coh(X, y)

    def load_best_models_with_gist_features(self):
        sg = gce.svm_gist()
        svm = sg.load_clf_svm()
        nn = sg.load_clf_nn()
        return svm, nn

    def plot_gist_svm_tuning(self, grid):
        ping.plot_gist_svm(grid)

    def plot_gist_nn_tuning(self, grid):
        ping.plot_gist_nn(grid)

    def load_best_models_with_coh_features(self):
        coh = cce.svm_coh()
        rf = coh.load_clf_rf()
        ab = coh.load_clf_rf()
        return rf, ab

    def plot_coh_rf_tuning(self, grid):
        ping.plot_coh_rf(grid)


    def plot_coh_ab_tuning(self, grid):
        ping.plot_coh_ab(grid)

    def eval_final(self):
        emogsv = eval.evaluate_model_on_gsv()
        # emogsv.extract_gist_for_images_and_save()
        # emogsv.read_gsv_gist_data()
        # emogsv.get_best_model_trained()
        emogsv.evaluate()

rm = run_main()
rm.main()

#ecf = ec.evaluate_classifiers()
#X, y, XT, yT = ecf.load_data(c.FEATURE_TYPE_LIST[0])
#print 'GIST', X.shape, ' ', y.shape, ' ', XT.shape, ' ', yT.shape
#X, y, XT, yT = ecf.load_data(c.FEATURE_TYPE_LIST[1])
#print 'COH', X.shape, ' ', y.shape, ' ', XT.shape, ' ', yT.shape
#X, y, XT, yT = ecf.load_data(c.FEATURE_TYPE_LIST[2])
#print 'ORB', X.shape, ' ', y.shape, ' ', XT.shape, ' ', yT.shape