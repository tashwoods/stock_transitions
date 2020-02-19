import sys, os, math, shutil, time, argparse, matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from pylab import *
from pandas.plotting import scatter_matrix
import seaborn as sns
from matplotlib.pyplot import cm
import multiprocessing
matplotlib.use("Agg") #disable python gui rocket in mac os dock
import matplotlib.pyplot as plt
from functools import partial

from classes import *

def make_test_train_datasets(file_name):
  #Check metadata of given stock
  print('-------------------------------------------------------')
  print('DATA FROM: {}'.format(file_name))
  formatted_data = get_data(file_name)
  print(formatted_data)
  #Extract train and test set
  train_set, test_set = train_test_split(formatted_data, test_size = args.test_size, random_state = 42) 

  #Order train and test set by ascending date
  train_set = train_set.sort_values(by = args.date_name)
  test_set = test_set.sort_values(by = args.date_name)
  if args.verbose > 1:
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
    if '/' in file_name:
      file_name=file_name[file_name.rfind('/')+1:-7].upper()
    else:
      file_name=file_name[:-7].upper()
  return file_name

def get_data(file_name):
  #reformat date from Y-M-D to Y.day/365
  formatted_data = pd.read_csv(file_name) 
  formatted_data[args.date_name] = formatted_data[args.date_name].str.replace('-','').astype(int)
  formatted_data[args.date_name] = formatted_data[args.date_name].apply(get_day_of_year)
  return formatted_data

def get_day_of_year(date):
  date = pd.to_datetime(date, format='%Y%m%d')
  first_day = pd.Timestamp(year=date.year, month=1, day=1)
  days_in_year = get_number_of_days_in_year(date.year)
  day_of_year = date.year+(((date - first_day).days)/(days_in_year)) 
  if args.verbose > 1:
    print('date: {} day of year: {} ndays_year: {}, calculated: {}'.format(date,((date - first_day).days), days_in_year, day_of_year))
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

def make_histograms(stock_object):
  for var in stock_object.train_set:
    make_hist(var, stock_object.train_set)

def multithread_plot(stock_object):
  pool = multiprocessing.Pool(args.max_number_processes)
  prod_x=partial(make_hist, dataset=stock_object.train_set) # prod_x has only one argument x (y is fixed to 10)
  result_list = pool.map_async(prod_x, stock_object.train_set.columns)
  pool.close()
  pool.join()

def make_hist(var, dataset):
  fig, ax = plt.subplots()
  ax.hist(dataset[var])
  plt.xlabel(var)
  plt.ylabel('Count')
  plt.title(var + ' Histogram')
  var_stats = str(dataset[var].describe())
  var_stats = var_stats[:var_stats.find('Name')]
  plt.text(0.75, 0.6, str(var_stats), transform = ax.transAxes)
  plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/' + var + '_hist.pdf')
  plt.close('all')



def worker_plots(stock_object):
  for var in stock_object.train_set.columns:
    stock_object.make_histograms(var)
  stock_object.make_overlay_plot()
  stock_object.make_scatter_plots()
  stock_object.make_scatter_heat_plots()
  stock_object.make_correlation_plots()
  stock_object.make_time_dependent_plots()

def make_histograms(self, var, output_dir):
  dataset = self.train_set
  fig, ax = plt.subplots()
  ax.hist(dataset[var])
  plt.xlabel(var)
  plt.ylabel('Count')
  plt.title(var + ' Histogram')
  var_stats = str(dataset[var].describe())
  var_stats = var_stats[:var_stats.find('Name')]
  plt.text(0.75, 0.6, str(var_stats), transform = ax.transAxes)
  plt.savefig(output_dir + '/' + self.stock_name + '/' + var + '_hist.pdf')
  plt.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for visualize.py')
  parser.add_argument('-f', '--input_file', type = str, help = 'text file with input file names')
  parser.add_argument('-v', '--verbose', dest = 'verbose', action = 'count', default = 0, help = 'Enable verbose output (not default). Add more vs for more output.')
  parser.add_argument('-t', '--test_size', type = float, dest = 'test_size', default = 0.2, help = 'Size of test set')
  parser.add_argument('-o', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  parser.add_argument('-d', '--date_name', type = str, dest = 'date_name', default = 'Date', help = 'name of Date variable in dataset')
  parser.add_argument('-vol', '--volume_name', type = str, dest = 'volume_name', default = 'Volume', help = 'name of Volume variable in dataset')
  parser.add_argument('-open_int', '--open_int_name', type = str, dest = 'open_int_name', default = 'OpenInt', help = 'name of OpenInt variable in dataset')
  parser.add_argument('-ncol', '--number_of_colors', type = int, dest = 'n_colors', default = 256, help = 'number of colors used in heatmaps')
  parser.add_argument('-colmin', '--minimum_color_value', type = int, dest = 'color_min', default = -1, help = 'minimum value of color map used in heatmaps')
  parser.add_argument('-colmax', '--maximum_color_value', type = int, dest = 'color_max', default = 1, help = 'maximum value of color map used in heatmaps')
  parser.add_argument('-palmin', '--minimum_pal_value', type = int, dest = 'pal_min', default = 20, help = 'minimum palette color value used in heatmaps')
  parser.add_argument('-palmax', '--maximum_pal_value', type = int, dest = 'pal_max', default = 220, help = 'maximum palette color value used in heatmaps')
  parser.add_argument('-indiv_plots', '--indiv_plots', type = int, dest = 'indiv_plots', default = 1, help = 'set to one to have individual stock plots')
  parser.add_argument('-overlay_stock_plots', '--overlay_stock_plots', type = int, dest = 'overlay_stock_plots', default = 0, help = 'set to one to have overlay stock plots')
  parser.add_argument('-max_number_processes', '--max_number_processes', type = int, dest = 'max_number_processes', default = multiprocessing.cpu_count(), help = 'maximum number of processes allowed to run')
  args = parser.parse_args()

  make_output_dir(args.output_dir)
  start_time = time.time()
  input_file = open(args.input_file, "r")
 
  stock_objects_list = list() 
  stock_objects_names = list() 

  for file_name in input_file:
    file_name = file_name.rstrip()
    if(os.stat(file_name).st_size) == 0:
      print('{} is empty, skipping this file'.format(file_name))
      continue
    make_nested_dir(args.output_dir, get_stock_name(file_name))
    test_set,train_set = make_test_train_datasets(file_name)
    stock_objects_list.append(stock_object_class(file_name, get_stock_name(file_name), test_set, train_set, args))
    stock_objects_names.append(get_stock_name(file_name))

  if args.indiv_plots == 1:
    pool = multiprocessing.Pool(args.max_number_processes)
    result_list = pool.map_async(worker_plots, stock_objects_list)
    pool.close()
    pool.join()

  if args.overlay_stock_plots == 1:
    #iterate over stock variables
    for var in stock_objects_list[0].train_set.columns:
      color = cm.rainbow(np.linspace(0,1,len(stock_objects_list)))
      for stock,c in zip(range(len(stock_objects_list)), color):
        if var != args.date_name:
          plt.plot('Date', var, data = stock_objects_list[stock].train_set, markerfacecolor = c, linewidth = 0.2, label = stock_objects_list[stock].stock_name)
          plt.xlabel('Date')
          plt.ylabel(var)
      plt.legend()
      plt.savefig(args.output_dir + '/stock_' + var + '_overlay.pdf')    
      plt.close('all')



  print('----- {} seconds ---'.format(time.time() - start_time))
