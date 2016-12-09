import lxml.etree as ET
import constants as c

labels = ['place_of_worship', 'fuel', 'parking', 'shop']
tree = ET.parse('../Resources/map_data/map_amherst_downtown.osm')
root = tree.getroot()
print "Parsing document: "
for tag in root.iter('tag'):
    if tag.attrib['k'] == 'amenity' and tag.attrib['v'] in labels:
        if tag.attrib['v'] == labels[0]:
           parent = tag.getparent()
           name_of_church = ''
           for sibling in parent:
             if sibling.tag == 'tag':
                 if sibling.attrib['k'] == 'name':
                     name = sibling.attrib['v']
                 if sibling.attrib['k'] == 'religion' and sibling.attrib['v'] == 'christian':
                     name_of_church = name
                     print name_of_church

