import lxml.etree as ET
import constants as c
import sys
import os
import csv

SIZE_GB = 10**7

class extract_osm_data:
    def __init__(self):
        self

    def read_and_parse_osm(self):
        print 'extract_osm_data: ', 'Start parsing...'
        context = ET.iterparse(c.PATH_OSM_DATA_AVAILABLE+c.FILENAME_OSM_DATA_AVAILABLE, tag='node')
        self.fast_iter(context, self.process_element)
        print 'extract_osm_data: ', 'All data saved successfully!!'

    def fast_iter(self, context, func, *args, **kwargs):
        """
        Studied http://stackoverflow.com/questions/7171140/using-python-iterparse-for-large-xml-files

        http://lxml.de/parsing.html#modifying-the-tree
        Based on Liza Daly's fast_iter
        http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
        See also http://effbot.org/zone/element-iterparse.htm
        """
        data = []
        if os.path.exists(c.PATH_OSM_DATA_SAVED):
            self.cleanup_osm_folder()
        for event, elem in context:
            data_row = func(elem, *args, **kwargs)
            if data_row:
                data.append(data_row)
                # if size gets bigger than 1GB save it in a file and free data
                print 'Current data size: ', sys.getsizeof(data)
                if sys.getsizeof(data) > SIZE_GB:
                    self.save_extracted_data_in_file(data)
                    #delete data after saving to a file
                    del data[:]

            # It's safe to call clear() here because no descendants will be
            # accessed

            elem.clear()

            # Also eliminate now-empty references from the root node to elem
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]

        self.save_extracted_data_in_file(data)
        del context

    def process_element(self, elem):
        #print "Parsing document: "
        is_amen_interest, label = self.is_amenity_of_interest(elem)
        if(is_amen_interest):
            data_row = self.get_data_row_to_save(elem, label)
            print data_row
            return data_row
        return []

    def is_amenity_of_interest(self, node):
        for tag in node:
            if tag.attrib['k'] == 'amenity' and tag.attrib['v'] in c.OSM_LABELS[1:]:
                return True, tag.attrib['v']
            elif tag.attrib['k'] == 'shop':
                return True, tag.attrib['k']
            elif tag.attrib['k'] == 'amenity' and tag.attrib['v'] == c.OSM_LABELS[0] and self.is_church(node):
                return True, tag.attrib['v']
        return False, ''

    def is_amenity(self, node):
        for tag in node:
            if tag.attrib['k'] == 'amenity':
                print tag.attrib['v']
                return True
        return False

    def is_church(self, node):
        for tag in node:
            if tag.attrib['k'] == 'religion' and tag.attrib['v'] == c.VAL_CHRISTIAN:
                return True
        return False

    def get_data_row_to_save(self, node, label):
        type, img_label, lat, lon, name, extra = label, '', c.FNAV, c.FNAV, c.NAV, c.NAV
        lat = node.attrib['lat']
        lon = node.attrib['lon']
        if(self.is_church(node)):
            type = c.OSM_LABELS[0]
            img_label = c.IMG_LABELS[0]
            for tag in node:
                if tag.attrib['k'] == 'name':
                    name = tag.attrib['v']
                elif tag.attrib['k'] == 'denomination':
                    extra = ' ' + tag.attrib['v']
        elif label == c.OSM_LABELS[1]:
            img_label = c.IMG_LABELS[1]
            for tag in node:
                if tag.attrib['k'] == 'name':
                    name = tag.attrib['v']
                elif tag.attrib['k'] == 'brand':
                    extra += ' ' + tag.attrib['v']
        elif label == c.OSM_LABELS[2]:
            img_label = c.IMG_LABELS[2]
            for tag in node:
                if tag.attrib['k'] == 'name':
                    name = tag.attrib['v']
                elif tag.attrib['k'] == 'operator':
                    extra += ' ' + tag.attrib['v']
                elif tag.attrib['k'] == 'designation':
                    extra += ' ' + tag.attrib['v']
        elif label == c.OSM_LABELS[3]:
            img_label = c.IMG_LABELS[3]
            for tag in node:
                if tag.attrib['k'] == 'name':
                    name = tag.attrib['v']
                elif tag.attrib['k'] == 'operator':
                    extra += ' ' + tag.attrib['v']
        else:
            print 'Wrong tag'

        return [type, img_label, lat, lon, name, extra]

    def save_extracted_data_in_file(self, data):
        if not os.path.exists(c.PATH_OSM_DATA_SAVED):
            os.makedirs(c.PATH_OSM_DATA_SAVED)
        name_ext = len(os.listdir(c.PATH_OSM_DATA_SAVED))
        filename = c.PATH_OSM_DATA_SAVED + c.FILENAME_OSM_DATA_SAVED + str(name_ext + 1) + '.csv'
        print 'extract_osm_data: ', 'Saved file: ', filename, '\n'
        with open(filename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(iter(data))

    def cleanup_osm_folder(self):
        folder = c.PATH_OSM_DATA_SAVED
        for file in os.listdir(folder):
            full_file = os.path.join(folder, file)
            try:
                if os.path.isfile(full_file):
                    os.unlink(full_file)
            except Exception as e:
                print e

    def read_saved_osm_data(self):
        print "\n"
        data = []
        if not os.path.exists(c.PATH_OSM_DATA_SAVED):
            os.makedirs(c.PATH_OSM_DATA_SAVED)
        num_files = len(os.listdir(c.PATH_OSM_DATA_SAVED))
        if num_files > 0:
            for i in range(num_files):
                filename = c.PATH_OSM_DATA_SAVED + c.FILENAME_OSM_DATA_SAVED + str(i + 1) + '.csv'
                print 'extract_osm_data: ', 'read_saved_osm_data: now reading ... ', filename
                with open(filename, 'rb') as f:
                    reader = csv.reader(f)
                    try:
                        for row in reader:
                            data.append(row)
                    except csv.Error as e:
                        print 'Couldn\'t read data for file %s line %d' % (filename, reader.line_num)
            return data
        else:
            print 'read_saved_osm_data: No data in the directory ', c.PATH_OSM_DATA_SAVED
            return []