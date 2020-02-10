import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

testsize = 0.2

def main():
  filename=sys.argv[1]
  diagnostic_plots(filename)
   
def diagnostic_plots(filename):
  #Check metadata of given stock
  stock_name = extract_stock_name(filename)
  print('-------------------------------------------------------')
  print('STOCK: {}'.format(stock_name))
  print('DATA FROM: {}'.format(filename))

  formatted_data = get_data(filename)
  formatted_data.info()
  print(formatted_data.head())

  #Extract train and test set (testsize is set at the beginning of visualize.py)
  train_set, test_set = train_test_split(formatted_data, test_size = testsize, random_state = 42) 
  print(train_set.info)
  print(test_set.info)

  
   
def extract_stock_name(filename):
  if filename.endswith('.us.txt'):
    filename=filename[:-7].upper()
  return filename
    

def get_data(filename):
  formatted_data = pd.read_csv(filename) 
  formatted_data.Date = formatted_data['Date'].str.replace('-','').astype(int)
  formatted_data['Date'] = formatted_data['Date'].apply(date_to_nth_day)
  return formatted_data

def date_to_nth_day(date):
    format='%Y%m%d'
    date = pd.to_datetime(date, format=format)
    new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
    modified_date = date.year+(((date - new_year_day).days + 1)/365)
    return modified_date


if __name__ == '__main__':
    main()
