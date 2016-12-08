NUMBER_OF_PAGES_CLASS_SCRAPPING = 20
CLASS_SCRAPPER_URL = "http://taginfo.openstreetmap.org/api/4/key/values?key=amenity&filter=all&lang=en&sortname=count&sortorder=desc&page=%d&rp=12&qtype=value&format=json_pretty"
THESAURUS_BASE = "http://thesaurus.altervista.org/thesaurus/v1"
THESAURUS_KEY = "unXYIXL2B0RpNA7V8XuG"
THESAURUS_LANG = "en_US"
THESAURUS_URL = THESAURUS_BASE + "?key=" + THESAURUS_KEY + "&word=%s" + "&language=" + THESAURUS_LANG + "&output=json"
PATH_FIGURES = '../Figures/'
PATH_DATA_HOME = '../Data/images/'
PATH_RESOURCES_HOME = '../Data/'
PATH_DATA_GIST_PICKLE = '../Data/pickle/gist'
PATH_DATA_COH_PICKLE = '../Data/pickle/coh'
PATH_DATA_ORB_PICKLE = '../Data/pickle/orb'
PATH_RES_GIST_TRAIN = '../Resources/gist/'
PATH_RES_COH_TRAIN = '../Resources/coh/'
PATH_RES_ORB_TRAIN = '../Resources/orb/'
DATA_TRAINING_GIST_PICKLED = '../Data/pickle/gist/training/training.p'
DATA_TEST_GIST_PICKLED = '../Data/pickle/gist/test/test.p'
DATA_TRAINING_COH_PICKLED = '../Data/pickle/coh/training/training.p'
DATA_TEST_COH_PICKLED = '../Data/pickle/coh/test/test.p'
DATA_TRAINING_ORB_PICKLED = '../Data/pickle/orb/training/training.p'
DATA_TEST_ORB_PICKLED = '../Data/pickle/orb/test/test.p'
FEATURE_TYPE_LIST = ['GIST','COH','ORB']
LABEL_MAP = {'shopfront': 3, 'parking_lot': 2, 'gas_station': 1, 'church': 0}