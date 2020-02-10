import sys, os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt





def diagnostic_plots(filename):
  #Check metadata of given stock
  stock_name = extract_stock_name(filename)
  print('-------------------------------------------------------')
  print('STOCK: {}'.format(stock_name))
  print('DATA FROM: {}'.format(filename))
  formatted_data = get_data(filename)

  #Extract train and test set (testsize is set at the beginning of visualize.py)
  train_set, test_set = train_test_split(formatted_data, test_size = args.testsize, random_state = 42) 

  if args.verbose > 0:
    formatted_data.info()
    print('Head of entire dataset')
    print(formatted_data.head())
    print('Training dataset')
    print(train_set.info)
    print('Test dataset')
    print(test_set.info)

    return train_set
  
def extract_stock_name(filename):
  if filename.endswith('.us.txt'):
    filename=filename[:-7].upper()
  return filename

def get_data(filename):
  #reformat date from Y-M-D to Y.day/365
  formatted_data = pd.read_csv(filename) 
  formatted_data.Date = formatted_data['Date'].str.replace('-','').astype(int)
  formatted_data['Date'] = formatted_data['Date'].apply(date_to_nth_day)
  return formatted_data

def date_to_nth_day(date):
  date = pd.to_datetime(date, format='%Y%m%d')
  new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
  days_in_year = number_of_days_in_year(date.year)
  modified_date = date.year+(((date - new_year_day).days)/(days_in_year)) 
  if args.verbose > 1:
    print('date: {} day of year: {} ndays_year: {}, calculated: {}'.format(date,((date - new_year_day).days), days_in_year, modified_date))
  return modified_date

def number_of_days_in_year(year):
  first_day = pd.Timestamp(year,1,1)
  last_day = pd.Timestamp(year,12,31)
  #Add 1 to days_in_year (e.g. 20181231 -->2018.99 and 20190101 --> 2019.00)
  number_of_days = (last_day - first_day).days + 1
  return number_of_days

def make_outputdirs(outputdir, filename):
  if not os.path.exists(outputdir):
    stock_name = extract_stock_name(filename)
    os.makedirs(outputdir + '/' + stock_name)
  else:
    print('Exiting. Outputdir, {}, already exists. Delete and rerun, rename, or run something else ;)')
    exit()

def make_plots(dataset):
  stock_name = extract_stock_name(args.filename)
  for var in dataset:
    print(var) 
    ax = dataset.hist(column = 'High') 
    fig = ax[0][0].get_figure()
    fig.savefig(args.outputdir + '/' + stock_name + '/' + var + '.pdf')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for visualize.py')
  parser.add_argument('-f', '--filename', type = str, help = 'input file')
  parser.add_argument('-v', '--verbose', dest = 'verbose', action = 'count', default = 0, help = 'Enable verbose output (not default). More vs provide more output.')
  parser.add_argument('-t', '--testsize', type = float, dest = 'testsize', default = 0.2, help = 'Size of test set')
  parser.add_argument('-o', '--outputdir', type = str, dest = 'outputdir', default = 'output', help = 'name of outputdir')
  args = parser.parse_args()

  make_outputdirs(args.outputdir, args.filename)
  train_set = diagnostic_plots(args.filename)
  make_plots(train_set)
