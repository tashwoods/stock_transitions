import sys, os, math, shutil
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from pylab import *
from pandas.plotting import scatter_matrix
import seaborn as sns
from matplotlib.pyplot import cm
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
    ax = stock_object.train_set.hist(column = var) 
    fig = ax[0][0].get_figure()
    ax[0][0].set_xlim(stock_object.train_set[var].min(), stock_object.train_set[var].max())
    var_stats = str(stock_object.train_set[var].describe())
    var_stats = var_stats[:var_stats.find('Name')]
    fig.text(0.7,0.5,str(var_stats))
    fig.savefig(args.output_dir + '/' + stock_object.stock_name + '/' + var + '_hist.pdf')
  plt.close('all')

def make_time_dependent_plots(stock_object):
  time_plots = list()
  for var in stock_object.train_set.columns:
    if var != args.date_name:
      ax = time_plots.append(stock_object.train_set.plot(x = args.date_name, y = var))
      plt.xlabel('Date')
      plt.ylabel(var)
      plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/' + var + '_time.pdf')
  plt.close('all')
  return time_plots
  

def make_overlay_plots(stock_object):
  for var in stock_object.train_set.columns:
    if var != args.date_name and var!= args.volume_name and var!= args.open_int_name:
      plt.plot('Date', var, data = stock_object.train_set, markerfacecolor = 'blue', linewidth = 0.2)
  plt.legend()
  plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/stock_variables_overlay.pdf')

def make_scatter_plots(stock_object):
  for i in range(len(stock_object.train_set.columns) - 1):
    for j in range(i+1, len(stock_object.train_set.columns) - 1):
      var1 = stock_object.train_set.iloc[:,i]
      var2 = stock_object.train_set.iloc[:,j]

      fig = stock_object.train_set.plot(kind = 'scatter', x = stock_object.train_set.columns[i], y = stock_object.train_set.columns[j], alpha = 0.1)
      plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/scatter_' + var1.name + '_' + var2.name + '.pdf')
      plt.close('all')

def make_scatter_heat_plots(stock_object):
  for i in range(len(stock_object.train_set.columns) - 1):
    for j in range(i+1, len(stock_object.train_set.columns) - 1):
      for k in range(j+1, len(stock_object.train_set.columns) - 1):
        for l in range(k+1, len(stock_object.train_set.columns) - 1):
          var1 = stock_object.train_set.iloc[:,i]
          var2 = stock_object.train_set.iloc[:,j]
          var3 = stock_object.train_set.iloc[:,k]
          var4 = stock_object.train_set.iloc[:,l]

          ax = stock_object.train_set.plot.scatter(x = stock_object.train_set.columns[i], y = stock_object.train_set.columns[j], c = stock_object.train_set.columns[l], colormap = 'viridis')
          plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/heat_scatter_' + var1.name + '_' + var2.name + '_' + var3.name + '_' + var4.name + '.pdf')
          plt.close('all')

def make_correlation_plots(stock_object):
  #simple correlation plot
  corr = stock_object.train_set.corr()
  axes = scatter_matrix(stock_object.train_set[['Date', 'Volume', 'Open', 'High', 'Low', 'Close']], alpha = 0.2)
  for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i,j].annotate("%.3f" %corr.iloc[i][j], (0.8, 0.8), xycoords = 'axes fraction', ha = 'center', va = 'center')
  plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/correlations_simple.pdf')
  plt.close('all')

  #heatmap correlation plot
  ax = sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, square = True)
  ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
  plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/correlations_heatmap.pdf')

  #heatmap with variable box size plot
  corr = pd.melt(corr.reset_index(), id_vars = 'index')
  corr.columns = ['x', 'y', 'value']
  make_complex_heatmap(corr['x'], corr['y'], size = corr['value'].abs())
  

def make_complex_heatmap(x, y, size):
  fig, ax = plt.subplots()

  #Map from column names to integer coordinates
  x_labels = [v for v in sorted(x.unique())]
  y_labels = [v for v in sorted(y.unique())]
  x_to_num = {p[1]:p[0] for p in enumerate(x_labels)}
  y_to_num = {p[1]:p[0] for p in enumerate(y_labels)}

  size_scale = 500

  palette = sns.diverging_palette(args.pal_min, args.pal_max, n = args.n_colors)

  plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
  ax = plt.subplot(plot_grid[:,:-1]) # Use the leftmost 14 columns of the grid for the main plot

  ax.scatter(x = x.map(x_to_num), y = y.map(y_to_num), s = size * size_scale, c = size.apply(value_to_color) , marker = 's')

  ax.set_xticks([x_to_num[v] for v in x_labels])
  ax.set_xticklabels(x_labels, rotation = 45, horizontalalignment = 'right')
  ax.set_yticks([y_to_num[v] for v in y_labels])
  ax.set_yticklabels(y_labels)

  ax.grid(False, 'major')
  ax.grid(True, 'minor')
  ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor = True)
  ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor = True)
  ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
  ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

  # Add color legend on the right side of the plot
  ax = plt.subplot(plot_grid[:,-1]) 
  col_x = [0]*len(palette) 
  bar_y=np.linspace(args.color_min, args.color_max, args.n_colors)
  bar_height = bar_y[1] - bar_y[0]
  ax.barh(y=bar_y,width=[5]*len(palette),height=bar_height,color=palette,linewidth=0)
  ax.set_xlim(1, 2) 
  ax.set_ylim(-1,1)
  ax.grid(False) 
  ax.set_facecolor('white')

  #Adjust ticks on correlation plot
  ax.set_xticks([]) # Remove horizontal ticks
  ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
  ax.yaxis.tick_right() # Show vertical ticks on the right

  plt.savefig(args.output_dir + '/' + stock_object.stock_name + '/correlations_heatmap_fancy.pdf')
 

def value_to_color(val):
  palette = sns.diverging_palette(args.pal_min, args.pal_max, n = args.n_colors)
  if math.isnan(val):
    val = 0
  val_position = float((val - args.color_min))/(args.color_max - args.color_min) 
  ind = int(val_position * (args.n_colors - 1))
  palette_ind = palette[ind]

  return palette_ind

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
  parser.add_argument('-indiv_plots', '--indiv_plots', type = int, dest = 'indiv_plots', default = 0, help = 'set to one to have individual stock plots')
  parser.add_argument('-overlay_stock_plots', '--overlay_stock_plots', type = int, dest = 'overlay_stock_plots', default = 1, help = 'set to one to have overlay stock plots')
  args = parser.parse_args()

  make_output_dir(args.output_dir)

  input_file = open(args.input_file, "r")
 
  stock_objects_list = list() 
  stock_objects_names = list() 

  for file_name in input_file:
    print(file_name)
    file_name = file_name.rstrip()
    if(os.stat(file_name).st_size) == 0:
      print('{} is empty, skipping this file'.format(file_name))
      continue
    make_nested_dir(args.output_dir, get_stock_name(file_name))
    test_set,train_set = make_test_train_datasets(file_name)
    stock_object = stock_object_class(file_name, get_stock_name(file_name), test_set, train_set)
    stock_objects_list.append(stock_object)
    stock_objects_names.append(get_stock_name(file_name))


  if args.overlay_stock_plots == 1:
    #iterate over stock variables
    for var in stock_objects_list[0].train_set.columns:
      color = cm.rainbow(np.linspace(0,1,len(stock_objects_list)))
      print('n: {}'.format(color))
      for stock,c in zip(range(len(stock_objects_list)), color):
        if var != args.date_name:
          plt.plot('Date', var, data = stock_objects_list[stock].train_set, markerfacecolor = c, linewidth = 0.2, label = stock_objects_list[stock].stock_name)
          plt.xlabel('Date')
          plt.ylabel(var)
      plt.legend()
      plt.savefig(args.output_dir + '/stock_' + var + '_overlay.pdf')    
      plt.close('all')

 
  if args.indiv_plots == 1:
    make_histograms(stock_object)
    make_overlay_plots(stock_object)
    make_scatter_plots(stock_object)
    make_scatter_heat_plots(stock_object)
    make_correlation_plots(stock_object)
    make_time_dependent_plots(stock_object)

    
