from imported_libraries import *

def add_attributes(dataset): #Natasha: make this less hardcoded and more dynamic
  dataset['Open_Close'] = dataset['Open']/dataset['Close']
  dataset['Low_High'] = dataset['Low']/dataset['High']
  return dataset

def make_test_train_datasets(file_name, args):
  #Check metadata of given stock
  formatted_data = get_data(file_name, args.date_name)
  if len(args.drop_columns) > 0:
    formatted_data = formatted_data.drop(args.drop_columns, axis = 1)

  if args.combined_features == 1:
    formatted_data = add_attributes(formatted_data)

  #get index of test date split before possibly scaling data, which will make this more difficult to do
 
  first_test_date = get_day_of_year(args.year_test_set + args.month_test_set + args.day_test_set)
  string_last_test_date = str(int(args.year_test_set) + 1 ) + '0101'
  print(string_last_test_date)
  last_test_date = get_day_of_year(str(int(args.year_test_set) + 1 ) + '0101') #Natasha this is hard-coded to only test over a year span
  print('last test date: {}'.format(last_test_date))
  first_test_index = (formatted_data[args.date_name] >= first_test_date).idxmax()
  last_test_index = (formatted_data[args.date_name] >= last_test_date).idxmax()
  print('first test index: {}'.format(first_test_index))
  print('last test index: {}'.format(last_test_index))
  print('first test value: {} '.format(formatted_data.loc[first_test_index]))
  print('last test value: {} '.format(formatted_data.loc[last_test_index]))

  if args.scale_features == 1:
    scaled_features = StandardScaler().fit_transform(formatted_data.values)
    formatted_data = pd.DataFrame(scaled_features, index = formatted_data.index, columns = formatted_data.columns)

  #Extract train and test set
  train_set = formatted_data[:first_test_index]
  test_set = formatted_data[first_test_index:]
  year_test_set = formatted_data[first_test_index:last_test_index]

  #Order train and test set by ascending date, likely not needed, but does not hurt
  train_set = train_set.sort_values(by = args.date_name)
  test_set = test_set.sort_values(by = args.date_name)
  year_test_set = year_test_set.sort_values(by = args.date_name)
  
  return test_set, train_set, year_test_set

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


