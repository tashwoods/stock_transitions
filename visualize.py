import sys, os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from classes import *


def make_test_train_datasets(file_name):
  #Check metadata of given stock
  print('-------------------------------------------------------')
  print('DATA FROM: {}'.format(file_name))
  formatted_data = get_data(file_name)

  #Extract train and test set
  train_set, test_set = train_test_split(formatted_data, test_size = args.test_size, random_state = 42) 

  if args.verbose > 0:
    formatted_data.info()
    print('Head of entire dataset')
    print(formatted_data.head())
    print('Training dataset')
    print(train_set.info)
    print('Test dataset')
    print(test_set.info)

    return test_set, train_set
  
def get_stock_name(file_name):
  if file_name.endswith('.us.txt'):
    file_name=file_name[:-7].upper()
  return file_name

def get_data(file_name):
  #reformat date from Y-M-D to Y.day/365
  formatted_data = pd.read_csv(file_name) 
  formatted_data.Date = formatted_data['Date'].str.replace('-','').astype(int)
  formatted_data['Date'] = formatted_data['Date'].apply(get_day_of_year)
  return formatted_data

def get_day_of_year(date):
  date = pd.to_datetime(date, format='%Y%m%d')
  first_day = pd.Timestamp(year=date.year, month=1, day=1)
  days_in_year = get_number_of_days_in_year(date.year)
  day_of_year = date.year+(((date - first_day).days)/(days_in_year)) 
  if args.verbose > 1:
    print('date: {} day of year: {} ndays_year: {}, calculated: {}'.format(date,((date - first_day).days), days_in_year, modified_date))
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
    print('Exiting. Outputdir, {}, already exists. Delete, rename, or run something else ;)')
    exit()

def make_nested_dir(output_dir, nested_dir):
 Path(output_dir + '/' + nested_dir).mkdir(parents=True, exist_ok=True) 

def make_histograms(stock_object):
  for var in stock_object.train_set:
    print(var) 
    ax = stock_object.train_set.hist(column = var) 
    fig = ax[0][0].get_figure()
    fig.savefig(args.output_dir + '/' + stock_object.stock_name + '/' + var + '.pdf')

def make_time_dependent_plots(stock_object):
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for visualize.py')
  parser.add_argument('-f', '--file_name', type = str, help = 'input file')
  parser.add_argument('-v', '--verbose', dest = 'verbose', action = 'count', default = 0, help = 'Enable verbose output (not default). Add more vs for more output.')
  parser.add_argument('-t', '--test_size', type = float, dest = 'test_size', default = 0.2, help = 'Size of test set')
  parser.add_argument('-o', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  args = parser.parse_args()

  make_output_dir(args.output_dir)

  make_nested_dir(args.output_dir, get_stock_name(args.file_name))
  test_set,train_set = make_test_train_datasets(args.file_name)
  stock_object = stock_object(args.file_name, get_stock_name(args.file_name), test_set, train_set)
  make_histograms(stock_object)
  make_time_dependent_plots(stock_object)
