import urllib
import json
import time
import constants as c

class extract_all_classes:

    def __init__(self):
        self

    def main(self):
        """
        Extracts data in JSON format from http://taginfo.openstreetmap.org/
        One data element is of type:
                {
              "value": "parking",
              "count": 2203761,
              "fraction": 0.2119,
              "in_wiki": true,
              "description": "A place for parking cars"
                }
        :return:
        """
        wiki_verified = []
        start_scraping_classes = time.time()

        total_num_classes = 0
        total_wiki_verified = 0
        for i in range(1, c.NUMBER_OF_PAGES_CLASS_SCRAPPING + 1):
            url = c.CLASS_SCRAPPER_URL % i
            response = urllib.urlopen(url)
            raw_data = json.loads(response.read())
            num_classes_current_page = len(raw_data["data"])
            for j in range(0, num_classes_current_page):
                total_num_classes += 1
                if raw_data["data"][j]["in_wiki"] == True:
                    total_wiki_verified += 1
                    wiki_verified.append(raw_data["data"][j]["value"])
                    print "extract_all_classes: " + raw_data["data"][j]["value"] + " " + str(raw_data["data"][j]["fraction"]) + " " + str(raw_data["data"][j]["count"])

        print "\n" + "extract_all_classes: " + "Total: " + str(total_num_classes)
        print "extract_all_classes: " + "Wiki Verified: " + str(total_wiki_verified) + "\n\n"

        end_scrapping_classes = time.time()
        time_scrapping_classes_s = end_scrapping_classes - start_scraping_classes
        time_scrapping_classes_m, time_scrapping_classes_s = divmod(time_scrapping_classes_s, 60)
        time_scrapping_classes_h, time_scrapping_classes_m = divmod(time_scrapping_classes_m, 60)
        print "extract_all_classes: " + "Time taken to process all classes: %d hours %02d mins %02d secs" % (time_scrapping_classes_h, time_scrapping_classes_m, time_scrapping_classes_s)

        return wiki_verified
