from imported_libraries import *
from classes import *
from organize_input_output import *

def worker_plots(stock_object):
  stock_object.make_overlay_plot()
  stock_object.make_scatter_heat_plots() #scatter plots with z axis coloring by third variable
  stock_object.make_correlation_plots()
  stock_object.make_time_dependent_plots()

  #Extra plots
  #for var in stock_object.train_set.columns:
    #stock_object.make_histograms(var)
  #stock_object.make_scatter_plots() #scatter plots without z axis coloring by third variable

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


  #Organize and format input and output
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
    test_set,train_set = make_test_train_datasets(file_name, args.test_size, args.date_name)
    stock_objects_list.append(stock_object_class(file_name, get_stock_name(file_name), test_set, train_set, args))
    stock_objects_names.append(get_stock_name(file_name))


  #Plot data
  if args.indiv_plots == 1:
    pool = multiprocessing.Pool(args.max_number_processes)
    result_list = pool.map_async(worker_plots, stock_objects_list)
    pool.close()
    pool.join()

  if args.overlay_stock_plots == 1:
    for var in stock_objects_list[0].train_set.columns: #iterate over stock variables
      color = cm.rainbow(np.linspace(0,1,len(stock_objects_list)))
      for stock,c in zip(range(len(stock_objects_list)), color): #iterate over stocks
        if var != args.date_name:
          plt.plot('Date', var, data = stock_objects_list[stock].train_set, markerfacecolor = c, linewidth = 0.2, label = stock_objects_list[stock].stock_name)
          plt.xlabel('Date')
          plt.ylabel(var)
      plt.legend()
      plt.savefig(args.output_dir + '/stock_' + var + '_overlay.pdf')    
      plt.close('all')



  print('----- {} seconds ---'.format(time.time() - start_time))
