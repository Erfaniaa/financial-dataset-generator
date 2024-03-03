PAIR_NAMES_LIST_WITH_SOURCE = [("BCHAIN/DIFF", "Value", "nasdaq-data-link"),
							   ("BCHAIN/HRATE", "Value", "nasdaq-data-link"),
							   ("BCHAIN/CPTRA", "Value", "nasdaq-data-link"),
							   ("BCHAIN/TOTBC", "Value", "nasdaq-data-link"),
							   ("BITFINEX/ETHBTC", "Last", "nasdaq-data-link"),
							   ("BITFINEX/LTCBTC", "Last", "nasdaq-data-link"),
							   ("^DJI", "Close", "yfinance"),
							   ("GC=F", "Close", "yfinance"),
							   ("SI=F", "Close", "yfinance"),
							   ("EURUSD=X", "Close", "yfinance"),
							   ("BITFINEX/ETHUSD", "Last", "nasdaq-data-link"),
							   ("ETH-USD", "Close", "yfinance"),
							   ("LTC-USD", "Close", "yfinance"),
							   ("BTC-USD", "Close", "yfinance")]
TARGET_PAIR_NAME_WITH_SOURCE = ("BITFINEX/ETHBTC", "Last", "nasdaq-data-link")
SMA_LENGTHS_LIST = []
APPLY_FLIP_AUGMENTATION = False
APPLY_NOISE_AUGMENTATION = True
AUGMENTATION_NOISE_INTERVAL = 0.1 / 100
TRAIN_DATASET_NEW_SIZE_COEFFICIENT = 5
START_TIME = "2016-04-01"
END_TIME = "2022-10-01"
FORECAST_DAYS = 10
USE_WMA_FOR_FORECAST_DAYS = True
NUMBER_OF_CANDLES = 50
TRAIN_CSV_FILE_PATH = "dataset_train.csv"
TEST_CSV_FILE_PATH = "dataset_test.csv"
PRED_CSV_FILE_PATH = "dataset_predict.csv"
TEST_SET_SIZE_RATIO = 0.25
CSV_DELIMITER = ","
API_KEY_FILE_PATH = "API_KEY"
