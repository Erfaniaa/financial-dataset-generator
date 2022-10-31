# Financial Dataset Generator

Easy-to-use dataset generator for applying machine learning on financial markets

## Features

- You can run it fast, and it is easy to use.
- There are no complexities and no database usage in this project. Even dependencies are a few.
- It is easy to modify and customize.
- This project generates practical datasets for data scientists.
- You can read the code for educational purposes.

## Run

1. Clone the repository.
2. Run `pip3 install -r requirements.txt`.
3. Put your [Nasdaq Data Link](https://data.nasdaq.com/) API key in the `API_KEY` file.
4. Run `python3 main.py`.

This will generate train set and test set for you.

## Config

To define the strategy, you can:

- Change `config.py` constants.
- Define new indicators in `indicators.py`.

## Config.py Description

- `QUOTES_LIST_WITH_SOURCE`: What's your machine learning model input?
- `TARGET_QUOTE_WITH_SOURCE`: What's your machine learning model output? 
- `SMA_LENGTHS_LIST`: Do you want to generate a dataset with some moving averages?
- `APPLY_FLIP_AUGMENTATION` and `APPLY_NOISE_AUGMENTATION`: Using data augmentations
- `AUGMENTATION_NOISE_INTERVAL`: Set the amount of augmentation noise
- `TRAIN_DATASET_NEW_SIZE_COEFFICIENT`: How much augmented data do you want?
- `START_TIME` and `END_TIME`: The time interval for the dataset
- `FORECAST_DAYS`: How many days is your target?
- `USE_WMA_FOR_FORECAST_DAYS`: Do you want to use linear weighted moving average for your target?
- `NUMBER_OF_CANDLES`: Number of candles your machine learning model needs as its input
- `TRAIN_CSV_FILE_PATH`, `TEST_CSV_FILE_PATH`, and `PREDICT_CSV_FILE_PATH`: Output CSV file paths
- `TEST_SET_SIZE_RATIO`: Test set size to whole dataset size ratio
- `CSV_DELIMITER`: The delimiter in every generated CSV file
- `API_KEY_FILE_PATH`: Path to the Nasdaq Data Link API key file

## See Also

- [Binance Futures Trading Bot](https://github.com/erfaniaa/binance-futures-trading-bot)
- [Binance Spot Trading Bot](https://github.com/smzerehpoush/binance-spot-trading-bot)

## Credits

[Erfan Alimohammadi](https://github.com/Erfaniaa) and [Amir Reza Shahmiri](https://github.com/Amirrezashahmiri)
