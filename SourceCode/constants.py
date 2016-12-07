NUMBER_OF_PAGES_CLASS_SCRAPPING = 20
CLASS_SCRAPPER_URL = "http://taginfo.openstreetmap.org/api/4/key/values?key=amenity&filter=all&lang=en&sortname=count&sortorder=desc&page=%d&rp=12&qtype=value&format=json_pretty"
THESAURUS_BASE = "http://thesaurus.altervista.org/thesaurus/v1"
THESAURUS_KEY = "unXYIXL2B0RpNA7V8XuG"
THESAURUS_LANG = "en_US"
THESAURUS_URL = THESAURUS_BASE + "?key=" + THESAURUS_KEY + "&word=%s" + "&language=" + THESAURUS_LANG + "&output=json"
PATH_DATA_HOME = '../Data/images/'
PATH_DATA_PICKLE = '../Data/pickle'
DATA_TRAINING_PICKLED = '../Data/pickle/training/training.p'
DATA_TEST_PICKLED = '../Data/pickle/test/test.p'

LABEL_MAP = {'shopfront': 3, 'parking_lot': 2, 'gas_station': 1, 'church': 0}