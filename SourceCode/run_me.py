import extract_all_classes as eac
import extract_images as ei
class run_main(object):
    def __init__(self):
        self

    def main(self):
        classes = eac.extract_all_classes().main()
        print "main: " + str(classes)
        ei.extract_images().main(classes)

rm = run_main()
rm.main()
