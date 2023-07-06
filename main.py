from tracemalloc import start
import config
import indicators

import nasdaqdatalink
import yfinance
import pandas as pd

import datetime
import random


def date_string_to_datetime(date_string):
	return datetime.datetime.strptime(date_string, "%Y-%m-%d")


def get_next_day_string(current_day_string):
	current_day_datetime = date_string_to_datetime(current_day_string)
	next_day_datetime = current_day_datetime + datetime.timedelta(days=1)
	next_day_string = next_day_datetime.strftime("%Y-%m-%d")
	return next_day_string


def date_range(start_datetime, end_datetime):
    for i in range(int((end_datetime - start_datetime).days) + 1):
        yield start_datetime + datetime.timedelta(i)


def fill_missing_data(df, start_time, end_time):
	start_datetime = date_string_to_datetime(start_time)
	end_datetime = date_string_to_datetime(end_time)
	df_new = df[0:0]
	dates_list = [str(datetime)[:10] for datetime in date_range(start_datetime, end_datetime)]
	datetimes_list = [datetime for datetime in date_range(start_datetime, end_datetime)]
	dates_index = 0
	first_df_row_value = df.iloc[0, 1]
	while dates_list[dates_index] < str(df.iloc[0, 0])[:10]:
		new_row = pd.DataFrame({"Date": [datetimes_list[dates_index]],
								"Value": [first_df_row_value]})
		df_new = pd.concat([df_new, new_row], axis=0, ignore_index=True)
		dates_index += 1
	last_dates_index = dates_index
	df_index = 0
	for dates_index in range(last_dates_index, len(dates_list)):
		if df_index < df.shape[0] and str(df.iloc[df_index, 0])[:10] == dates_list[dates_index]:
			new_row = pd.DataFrame({"Date": [df.iloc[df_index, 0]],
									"Value": [df.iloc[df_index, 1]]})
			df_new = pd.concat([df_new, new_row], axis=0, ignore_index=True)
			last_new_row_value = df.iloc[df_index, 1]
			df_index += 1
		elif df_index >= df.shape[0] or str(df.iloc[df_index, 0])[:10] > dates_list[dates_index]:
			new_row = pd.DataFrame({"Date": [datetimes_list[dates_index]],
									"Value": [last_new_row_value]})
			df_new = pd.concat([df_new, new_row], axis=0, ignore_index=True)
			if df_index >= df.shape[0]:
				print("WARNING: Missing data added to the end of the downloaded dataframe. Date:", datetimes_list[dates_index])
	return df_new


def remove_extra_data(df, start_time, end_time):
	rows_to_be_removed = []
	for index, row in df.iterrows():
		if str(row["Date"])[:10] < start_time or str(row["Date"])[:10] > end_time:
			rows_to_be_removed.append(index)
	df = df.drop(rows_to_be_removed)
	return df


def get_data_for_quote(quote, column_name, data_source, start_time, end_time, forecast_days):
	if data_source == "nasdaq-data-link":
		df = nasdaqdatalink.get(quote, start_date=start_time, end_date=end_time)
	elif data_source == "yfinance":
		df = yfinance.download(quote, start=get_next_day_string(start_time), end=get_next_day_string(end_time), interval="1d", auto_adjust=True, prepost=True, threads=True)
	df = df.reset_index()
	df = df[['Date', column_name]]
	df = df.rename(columns={column_name: "Value"})
	return df


def get_values_list_from_dataframe(df):
	return df["Value"].tolist()


def get_values_for_quotes_list(quotes_with_column_name_list, start_time, end_time, forecast_days):
	all_quotes_values_list = []
	for quote, column_name, data_source in quotes_with_column_name_list:
		print("Quote:", quote, "(" + str(column_name) + ", " + str(data_source) + ")")
		df = get_data_for_quote(quote, column_name, data_source, start_time, end_time, forecast_days)
		print("Data downloaded")
		df = remove_extra_data(df, start_time, end_time)
		print("Extra data removed")
		df = fill_missing_data(df, start_time, end_time)
		print("Missing data filled")
		values_list = get_values_list_from_dataframe(df)
		print("_" * 80)
		all_quotes_values_list.append(values_list)
	return all_quotes_values_list


def get_target_quote_list(quote, column_name, data_source, start_time, end_time, forecast_days, number_of_candles, use_wma_for_forecast_days):
	if data_source == "nasdaq-data-link":
		df = nasdaqdatalink.get(quote, start_date=start_time, end_date=end_time)
	elif data_source == "yfinance":
		df = yfinance.download(quote, start=get_next_day_string(start_time), end=get_next_day_string(end_time), interval="1d", auto_adjust=True, prepost=True, threads=True)
	df = df.reset_index()
	df = remove_extra_data(df, start_time, end_time)
	target_list = []
	close_prices = []
	for index, row in df.iterrows():
		close_prices.append(float(row[column_name]))
	for i in range(len(close_prices)):
		if i >= number_of_candles - 1 and i + 1 + forecast_days <= len(close_prices):
			if use_wma_for_forecast_days:
				if close_prices[i] <= indicators.get_wma(close_prices[i + 1:i + 1 + forecast_days]):
					target_list.append(1)
				else:
					target_list.append(0)
			else:
				if close_prices[i] <= close_prices[i + forecast_days]:
					target_list.append(1)
				else:
					target_list.append(0)
		close_prices.append(float(row[column_name]))
	return target_list


def concat_no_target_dataset_with_targets_list(no_target_dataset, targets_list):
	dataset = no_target_dataset.copy()
	for i in range(min(len(dataset), len(targets_list))):
		dataset[i].append(targets_list[i])
	return dataset


def generate_no_target_dataset_from_quotes_values_list(quotes_values_list, number_of_candles, sma_lengths_list):
	no_target_dataset = []
	for i in range(len(quotes_values_list[0]) - number_of_candles):
		no_target_dataset_row = []
		for j in range(len(quotes_values_list)):
			current_values = quotes_values_list[j][i:i + number_of_candles]
			no_target_dataset_row.extend(current_values)
			for k in range(len(sma_lengths_list)):
				sma_list = [indicators.get_average(current_values[t - sma_lengths_list[k]:t]) for t in range(sma_lengths_list[k], len(current_values))]
				no_target_dataset_row.extend(sma_list)
		no_target_dataset.append(no_target_dataset_row)
	return no_target_dataset


def split_train_and_test_dataset(dataset, test_set_size_ratio):
	dataset_train = dataset[:int((1 - test_set_size_ratio) * len(dataset))]
	dataset_test = dataset[int((1 - test_set_size_ratio) * len(dataset)):]
	return dataset_train, dataset_test


def add_noise_to_dataset(dataset_train, augmentation_noise_interval, train_dataset_new_size_coefficient):
	dataset_train_new = []
	for row in dataset_train:
		dataset_train_new.append(row)
		for i in range(train_dataset_new_size_coefficient - 1):
			new_row = []
			for value in row[:-1]:
				r = 1 + (2 * (random.random() - 0.5) * augmentation_noise_interval)
				new_value = value * r
				new_row.append(new_value)
			new_row.append(row[-1])
			dataset_train_new.append(row)
	return dataset_train_new


def save_dataset_to_file(dataset_train, dataset_test, dataset_pred, train_csv_file_path, test_csv_file_path, pred_csv_file_path, csv_delimiter):
	with open(train_csv_file_path, 'w') as file:
		for dataset_line in dataset_train:
			for index, data in enumerate(dataset_line):
				file.write(str(data))
				if index != len(dataset_line) - 1:
					file.write(csv_delimiter)
			file.write('\n')
	with open(test_csv_file_path, 'w') as file:
		for dataset_line in dataset_test:
			for index, data in enumerate(dataset_line):
				file.write(str(data))
				if index != len(dataset_line) - 1:
					file.write(csv_delimiter)
			file.write('\n')
	with open(pred_csv_file_path, 'w') as file:
		for dataset_line in dataset_pred:
			for index, data in enumerate(dataset_line):
				file.write(str(data))
				if index != len(dataset_line) - 1:
					file.write(csv_delimiter)
			file.write('\n')
	print("Train set size:", str(len(dataset_train)) + "x" + str(len(dataset_train[0])))
	print("Test set size:", str(len(dataset_test)) + "x" + str(len(dataset_test[0])))
	print("Dataset saved to file")
	print("_" * 80)


def concat_datasets(dataset1, dataset2):
	ret = []
	for row in dataset1:
		ret.append(row)
	for row in dataset2:
		ret.append(row)
	return ret


def remove_intersected_rows_in_train_dataset(dataset_train, forecast_days):
	ret = []
	for i in range(len(dataset_train)):
		if i % forecast_days == 0:
			ret.append(dataset_train[i])
	return ret


def generate_dataset(quotes_list, target_quote_with_source, start_time, end_time, forecast_days, number_of_candles, train_csv_file_path, test_csv_file_path, pred_csv_file_path, csv_delimiter, test_set_size_ratio, sma_lengths_list, apply_noise_augmentation, augmentation_noise_sigma, train_dataset_new_size_coefficient, use_wma_for_forecast_days):
	all_quotes_values_list = get_values_for_quotes_list(quotes_list, start_time, end_time, forecast_days)
	no_target_dataset = generate_no_target_dataset_from_quotes_values_list(all_quotes_values_list, number_of_candles, sma_lengths_list)
	dataset_pred = no_target_dataset.copy()[-60:]
	print("Target quote:", target_quote_with_source[0], "(" + str(target_quote_with_source[1]) + ")")
	target_quote_list = get_target_quote_list(target_quote_with_source[0], target_quote_with_source[1], target_quote_with_source[2], start_time, end_time, forecast_days, number_of_candles, use_wma_for_forecast_days)
	dataset = concat_no_target_dataset_with_targets_list(no_target_dataset, target_quote_list)
	dataset_train, dataset_test = split_train_and_test_dataset(dataset, test_set_size_ratio)
	dataset_train = remove_intersected_rows_in_train_dataset(dataset_train, forecast_days)
	print("_" * 80)
	
	if apply_noise_augmentation:
		dataset_train = add_noise_to_dataset(dataset_train, augmentation_noise_sigma, train_dataset_new_size_coefficient)
		print("Noise augmentation applied")
		print("_" * 80)

	save_dataset_to_file(dataset_train, dataset_test, dataset_pred, train_csv_file_path, test_csv_file_path, pred_csv_file_path, csv_delimiter)


def read_api_key(api_key_file_path):
	nasdaqdatalink.read_key(filename=api_key_file_path)


def main():
	read_api_key(config.API_KEY_FILE_PATH)
	generate_dataset(config.QUOTES_LIST_WITH_SOURCE,
					 config.TARGET_QUOTE_WITH_SOURCE,
					 config.START_TIME,
					 config.END_TIME,
					 config.FORECAST_DAYS,
					 config.NUMBER_OF_CANDLES,
					 config.TRAIN_CSV_FILE_PATH,
					 config.TEST_CSV_FILE_PATH,
					 config.PRED_CSV_FILE_PATH,
					 config.CSV_DELIMITER,
					 config.TEST_SET_SIZE_RATIO,
					 config.SMA_LENGTHS_LIST,
					 config.APPLY_NOISE_AUGMENTATION,
					 config.AUGMENTATION_NOISE_INTERVAL,
					 config.TRAIN_DATASET_NEW_SIZE_COEFFICIENT,
					 config.USE_WMA_FOR_FORECAST_DAYS)


if __name__ == "__main__":
	main()
