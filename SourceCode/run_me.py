import extract_all_classes as eac
import extract_images as ei
import feature_extraction_GIST as gist
import feature_extraction_COH as coh
import feature_extraction_ORB as orb
import Evaluate_Classifiers as ec
import time
import constants as c
import plotting as ping



class run_main(object):

    def __init__(self):
        self

    def main(self):
    #    #All class labels processing
    #    classes = eac.extract_all_classes().main()
    #    #Extract data
    #    ei.extract_images().main(classes)
    #    self.extract_and_pickle_gist()
    #    self.extract_and_pickle_coh()
    #    self.extract_and_pickle_orb()
    #
    #    #Evaluate all classifiers
    #    self.eval_all_classifiers()

        #Plot all combined plot
        self.plot_all_combined()


    def extract_and_pickle_gist(self):
        # Pickle data
        start_pickling_time = time.time()

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

rm = run_main()
rm.main()
