from imported_libraries import *
from classes import *


def make_test_train_datasets(file_name, args):
  #Check metadata of given stock
  print('-------------------------------------------------------')
  print('DATA FROM: {}'.format(file_name))
  formatted_data = get_data(file_name, args.date_name)
  formatted_data = formatted_data.drop(args.drop_columns, axis = 1)
  print(formatted_data)
  #Extract train and test set
  train_set, test_set = train_test_split(formatted_data, test_size = args.test_size, random_state = 42) 

  #Order train and test set by ascending date
  train_set = train_set.sort_values(by = args.date_name)
  test_set = test_set.sort_values(by = args.date_name)

  return test_set, train_set
  
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


