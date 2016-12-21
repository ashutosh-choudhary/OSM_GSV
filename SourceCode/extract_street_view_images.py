import extract_osm_data as eod
import constants as c
import urllib2
import urllib
import os
import json


class extract_street_view_images:
    def __init__(self):
        self

    def extract_and_save_images(self):
        data = self.load_data_from_csv()
        num_images = len(data)
        for i in range(num_images):
            for heading in range(0, 301, 60):
                meta_url = c.GOOGLE_STREET_VIEW_META % (float(data[i][2]), float(data[i][3]), heading, c.GOOGLE_DEVELOPER_KEY)
                url = c.GOOGLE_STREET_VIEW_API % (float(data[i][2]), float(data[i][3]), heading, c.GOOGLE_DEVELOPER_KEY)
                try:
                    status_check = urllib2.urlopen(meta_url)
                    if not status_check['status'] == 'OK':
                        continue
                    response = urllib2.urlopen(url)
                except urllib2.HTTPError, e:
                    print "extract_images: Error: Couldn't open url " + url + " due to error code " + str(e.code)
                else:
                    rel_path = c.PATH_IMAGES_GSV
                    local_path = rel_path + str(data[i][2]) + 'x' + str(data[i][3])
                    file_name = data[i][1] + '_' + str(heading) + '.jpg'
                    if not os.path.exists(local_path):
                        os.makedirs(local_path)
                    full_filename = os.path.join(local_path, file_name)
                    print "extract_images: Downloading image " + url

                    urllib.urlretrieve(url, full_filename)

    def load_data_from_csv(self):
        ed = eod.extract_osm_data()
        data = ed.read_saved_osm_data()
        return data

