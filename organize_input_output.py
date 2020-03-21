from imported_libraries import *

def add_attributes(dataset): #Natasha: make this less hardcoded and more dynamic
  #dataset['Open_Close'] = dataset['Open']/dataset['Close']
  #dataset['Low_High'] = dataset['Low']/dataset['High']
  dataset['Close_Open_Change'] = (dataset['Close'] - dataset['Open'])/dataset['Open']
  dataset['High_Open_Change'] = (dataset['High'] - dataset['Open'])/dataset['Open']
  dataset['Low_Open_Change'] = (dataset['Open'] - dataset['Low'])/dataset['Open']
  return dataset

def averaged_dataframe(dataset, days):
  dfs = list()
  indices = np.arange(0, len(dataset.index), days)
  for i in range(len(indices) - 1):
    this_dataframe = dataset.iloc[indices[i]:indices[i+1]]
    dfs.append(this_dataframe.mean(axis=0))
  combined_dataframe = pd.concat(dfs,axis=1).T
  return combined_dataframe

def averaged_dataframe_array(dataset, days):
  split_data = [dataset[i:i+days] for i in range(0,dataset.shape[0],days)]
  return split_data

def make_test_train_datasets(file_name, args):
  #Check metadata of given stock
  formatted_data_unscaled = get_data(file_name, args.date_name)
  if len(args.drop_columns) > 0:
    formatted_data_unscaled = formatted_data_unscaled.drop(args.drop_columns, axis = 1)
  if args.combined_features == 1:
    formatted_data_unscaled = add_attributes(formatted_data_unscaled)

  #Predict_var at end of dataframe to make using StandardScaler easier later
  columns = list(formatted_data_unscaled.columns)
  columns.remove(args.predict_var)
  columns.append(args.predict_var)
  formatted_data_unscaled = formatted_data_unscaled[columns]

  #Get indicies for test and train sets
  first_test_date = get_day_of_year(args.year_test_set + args.month_test_set + args.day_test_set)
  string_last_test_date = str(int(args.year_test_set) + 1 ) + '0101'
  last_test_date = get_day_of_year(str(int(args.year_test_set) + 1 ) + '0101') #Natasha this is hard-coded to only test over a year span
  first_test_index = (formatted_data_unscaled[args.date_name] >= first_test_date).idxmax()
  last_test_index = (formatted_data_unscaled[args.date_name] >= last_test_date).idxmax()

  #Extract train and test set
  train_set_unscaled = formatted_data_unscaled[:first_test_index]
  test_set_unscaled = formatted_data_unscaled[first_test_index:]

  #Average test set over user defined days_in_week if test_set_averaged true
  if args.test_set_averaged:
    test_set_unscaled = averaged_dataframe(test_set_unscaled, args.days_in_week)

  #Order test and train set by date
  train_set_unscaled = train_set_unscaled.sort_values(by = args.date_name)
  test_set_unscaled = test_set_unscaled.sort_values(by = args.date_name)
  
  return test_set_unscaled, train_set_unscaled, formatted_data_unscaled

def get_stock_name(file_name):
  if file_name.endswith('.us.txt'):
    if '/' in file_name:
      file_name=file_name[file_name.rfind('/')+1:-7].upper()
    else:
      file_name=file_name[:-7].upper()
  return file_name

def get_data(file_name, date_name):
  #reformat date from Y-M-D to Y.day/365
  formatted_data = pd.read_csv(file_name) 
  formatted_data[date_name] = formatted_data[date_name].str.replace('-','').astype(int)
  formatted_data[date_name] = formatted_data[date_name].apply(get_day_of_year)
  return formatted_data

def get_day_of_year(date):
  date = pd.to_datetime(date, format='%Y%m%d')
  first_day = pd.Timestamp(year=date.year, month=1, day=1)
  days_in_year = get_number_of_days_in_year(date.year)
  day_of_year = date.year+(((date - first_day).days)/(days_in_year)) 
  return day_of_year

def get_number_of_days_in_year(year):
  first_day = pd.Timestamp(year,1,1)
  last_day = pd.Timestamp(year,12,31)
  #Add 1 to days_in_year (e.g. 20181231 -->2018.99 and 20190101 --> 2019.00)
  number_of_days = (last_day - first_day).days + 1
  return number_of_days

def make_output_dir(output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  else:
    make_dir_again = input("Outputdir: {} already exists. Delete it (y/n)?".format(output_dir))
    if make_dir_again == 'y':
      shutil.rmtree(output_dir)
      os.makedirs(output_dir)
    else:
      print('Delete/rename {} or run something else ;). Exiting.'.format(output_dir))
      exit()
    return

def make_nested_dir(output_dir, nested_dir):
 Path(output_dir + '/' + nested_dir).mkdir(parents=True, exist_ok=True) 


