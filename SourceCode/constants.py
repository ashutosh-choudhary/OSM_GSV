NUMBER_OF_PAGES_CLASS_SCRAPPING = 20
CLASS_SCRAPPER_URL = "http://taginfo.openstreetmap.org/api/4/key/values?key=amenity&filter=all&lang=en&sortname=count&sortorder=desc&page=%d&rp=12&qtype=value&format=json_pretty"

THESAURUS_BASE = "http://thesaurus.altervista.org/thesaurus/v1"
THESAURUS_KEY = "unXYIXL2B0RpNA7V8XuG"
THESAURUS_LANG = "en_US"
THESAURUS_URL = THESAURUS_BASE + "?key=" + THESAURUS_KEY + "&word=%s" + "&language=" + THESAURUS_LANG + "&output=json"

GOOGLE_DEVELOPER_KEY = 'AIzaSyBeZqRiR3nPq63S8UO2jea40azGhqUepk4'
GOOGLE_STREET_VIEW_META = "https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&location=%f,%f&heading=%d&pitch=10&key=%s"
GOOGLE_STREET_VIEW_API =  "https://maps.googleapis.com/maps/api/streetview?size=640x640&location=%f,%f&heading=%d&pitch=10&key=%s"


PATH_FIGURES = '../Figures/'
PATH_DATA_HOME = '../Data/images/'
PATH_IMAGES_GSV = '../Data/gsv/images/'
PATH_DATA_GSV = '../Data/gsv/data/'
PATH_MODEL_HOME = '../Data/results/'
PATH_DATA_GIST_PICKLE = '../Data/pickle/gist'
PATH_DATA_COH_PICKLE = '../Data/pickle/coh'
PATH_DATA_ORB_PICKLE = '../Data/pickle/orb'
PATH_RES_GIST_TRAIN = '../Resources/gist/'
PATH_RES_COH_TRAIN = '../Resources/coh/'
PATH_RES_ORB_TRAIN = '../Resources/orb/'

#This absolute path is being used only because the data size is
#huge i.e 185GB and would take a lot of time to copy to relative path
#the object of interest created with the entire north_america should be
#used
#PATH_OSM_DATA_AVAILABLE = '/Users/ashutoshchoudhary/Downloads/'
#FILENAME_OSM_DATA_AVAILABLE = 'north-america-latest.osm'
PATH_OSM_DATA_AVAILABLE = '../Resources/map_data/'
FILENAME_OSM_DATA_AVAILABLE = 'map_amherst_downtown.osm'

PATH_OSM_DATA_SAVED = '../Data/osm/'
FILENAME_OSM_DATA_SAVED = 'north-america-extracted'


DATA_TRAINING_GIST_PICKLED = '../Data/pickle/gist/training/training.p'
DATA_TEST_GIST_PICKLED = '../Data/pickle/gist/test/test.p'
DATA_TRAINING_COH_PICKLED = '../Data/pickle/coh/training/training.p'
DATA_TEST_COH_PICKLED = '../Data/pickle/coh/test/test.p'
DATA_TRAINING_ORB_PICKLED = '../Data/pickle/orb/training/training.p'
DATA_TEST_ORB_PICKLED = '../Data/pickle/orb/test/test.p'

FEATURE_TYPE_LIST = ['GIST','COH','ORB']
LABEL_MAP = {'shopfront': 3, 'parking_lot': 2, 'gas_station': 1, 'church': 0}
OSM_LABELS = ['place_of_worship', 'fuel', 'parking', 'shop']
IMG_LABELS = ['church', 'gas_station', 'parking_lot', 'shopfront']
#IMG_LABELS = range(4)
VAL_CHRISTIAN = 'christian'
NAV = ''
FNAV = -1 * float('inf')


FILENAME_AMHERST_GSV = 'gsv_amherst.p'
FILENAME_AMHERST_HEADING_VALS_GSV = 'true_headings_amherst.csv'
FILENAME_BEST_SVM_COARSE = 'best_svm_gist_coarse.pkl'
FILENAME_BEST_RF_COARSE = 'best_rf_coh_coarse.pkl'
FILENAME_BEST_AB_COARSE = 'best_ab_coh_coarse.pkl'
FILENAME_BEST_SVM_FINE = 'best_svm_gist_fine.pkl'
FILENAME_BEST_NN_COARSE = 'best_nn_gist_coarse.pkl'


TAG_EQUIVALENCE= {'parking': 'parking lot',
                  'place_of_worship':'church outdoor',
                  'school':'schoolhouse',
                  'restaurant':'restaurant',
                  'fuel':'gas station',
                  'bank':'bank outdoor',
                  'pharmacy':'pharmacy',
                  'hospital':'hospital',
                  'pub':'pub outdoor',
                  'bar':'bar',
                  'police':'police station',
                  'fire_station':'fire station'
                  }


