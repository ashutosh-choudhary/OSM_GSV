import extract_all_classes as eac
import extract_images as ei
import feature_extraction_GIST as gist
import model_estimation as estimate
import time

class run_main(object):
    def __init__(self):
        self

    def main(self):
    #    #All class labels processing
    #    classes = eac.extract_all_classes().main()
    #    #Extract data
    #    ei.extract_images().main(classes)
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

        me = estimate.model_estimation()
        me.base_main()

rm = run_main()
rm.main()
