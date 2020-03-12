from imported_libraries import *

def worker_plots(stock_object):
  #stock_object.make_overlay_plot() #overlay stock open, high, low, close vs. t plots
  #stock_object.make_scatter_heat_plots() #scatter plots with z axis coloring by third variable
  #stock_object.make_correlation_plots()
  stock_object.make_time_dependent_plots() #for each stock variable individually




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for visualize.py')
  parser.add_argument('-f', '--input_file', type = str, help = 'text file with input file names')
  parser.add_argument('-v', '--verbose', dest = 'verbose', action = 'count', default = 0, help = 'Enable verbose output (not default). Add more vs for more output.')
  parser.add_argument('-t', '--test_size', type = float, dest = 'test_size', default = 0.2, help = 'Size of test set')
  parser.add_argument('-o', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  parser.add_argument('-d', '--date_name', type = str, dest = 'date_name', default = 'Date', help = 'name of Date variable in dataset')
  parser.add_argument('-p', '--predict_variable_name', type = str, dest = 'predict_var', default = 'Close', help = 'name of variable in dataset that will be modelled')
  parser.add_argument('-vol', '--volume_name', type = str, dest = 'volume_name', default = 'Volume', help = 'name of Volume variable in dataset')
  parser.add_argument('-open_int', '--open_int_name', type = str, dest = 'open_int_name', default = 'OpenInt', help = 'name of OpenInt variable in dataset')
  parser.add_argument('-open', '--open_name', type = str, dest = 'open_name', default = 'Open', help = 'name of Open variable in dataset')
  parser.add_argument('-close', '--close_name', type = str, dest = 'close_name', default = 'Close', help = 'name of Close variable in dataset')
  parser.add_argument('-low', '--low_name', type = str, dest = 'low_name', default = 'Low', help = 'name of Low variable in dataset')
  parser.add_argument('-high', '--high_name', type = str, dest = 'high_name', default = 'High', help = 'name of High variable in dataset')
  parser.add_argument('-ncol', '--number_of_colors', type = int, dest = 'n_colors', default = 256, help = 'number of colors used in heatmaps')
  parser.add_argument('-colmin', '--minimum_color_value', type = int, dest = 'color_min', default = -1, help = 'minimum value of color map used in heatmaps')
  parser.add_argument('-colmax', '--maximum_color_value', type = int, dest = 'color_max', default = 1, help = 'maximum value of color map used in heatmaps')
  parser.add_argument('-palmin', '--minimum_pal_value', type = int, dest = 'pal_min', default = 20, help = 'minimum palette color value used in heatmaps')
  parser.add_argument('-palmax', '--maximum_pal_value', type = int, dest = 'pal_max', default = 220, help = 'maximum palette color value used in heatmaps')
  parser.add_argument('-indiv_plots', '--indiv_plots', type = int, dest = 'indiv_plots', default = 0, help = 'set to one to have individual stock plots')
  parser.add_argument('-overlay_stock_plots', '--overlay_stock_plots', type = int, dest = 'overlay_stock_plots', default = 1, help = 'set to one to have overlay stock plots')
  parser.add_argument('-max_number_processes', '--max_number_processes', type = int, dest = 'max_number_processes', default = multiprocessing.cpu_count(), help = 'maximum number of processes allowed to run')
  parser.add_argument('-drop_columns', '--drop_columns', nargs = '*', dest = 'drop_columns', default = [], help = 'list of columns to exclude from dataset')
  parser.add_argument('-N', '--number_files', type = int, dest = 'number_files', default = -1, help = 'number of files to randomly select from input file, if not specified or -1 all inputs files in input text file will be used')
  parser.add_argument('-min_file_size', '--min_file_size', type = int, dest = 'min_file_size', default = 100, help = 'minimum stock file size that will be used. This helps ignore empty files or files with few datapoints')
  parser.add_argument('-scale_features', '--scale_features', type = int, dest = 'scale_features', default = 1, help = 'set to one to scale features using StandardScaler(), 0 to not')
  parser.add_argument('-combined_features', '--combined_features', type = int, dest = 'combined_features', default = 0, help = 'set to one to add combined features to dataset, zero to not')
  parser.add_argument('-anticipated_columns', '--anticipated_columns', type = str, dest = 'anticipated_columns', default = 'Date,Open,High,Low,Close,Volume,OpenInt', help = 'list of columns that are expected in text files')
  parser.add_argument('-poly_reg', '--poly_reg', type = int, dest = 'poly_reg', default = 0, help = 'set to one to model stock open price with polynominal regression')
  parser.add_argument('-min_year', '--min_year', type = str, dest = 'min_year', default = '2008', help = 'first year to require stock data for')
  parser.add_argument('-max_year', '--max_year', type = str, dest = 'max_year', default = '2016', help = 'last year to require stock data for')
  parser.add_argument('-min_month', '--min_month', type = str, dest = 'min_month', default = '01', help = 'month of the first year to require stock data for')
  parser.add_argument('-max_month', '--max_month', type = str, dest = 'max_month', default = '01', help = 'month of the last year to require stock data for')
  parser.add_argument('-year_test_set', '--year_test_set', type = str, dest = 'year_test_set', default = '2016', help = 'year to begin test set with')
  parser.add_argument('-month_test_set', '--month_test_set', type = str, dest = 'month_test_set', default = '01', help = 'month to begin test set with')
  parser.add_argument('-day_test_set', '--day_test_set', type = str, dest = 'day_test_set', default = '01', help = 'day of month to begin test set with')
  parser.add_argument('-days_in_week', '--days_in_week', type = int, dest = 'days_in_week', default = 7, help = 'days in week to use for weekly averging')
  parser.add_argument('-days_in_month', '--days_in_month', type = int, dest = 'days_in_month', default = 30, help = 'days in month to use for month averging')
  parser.add_argument('-n_hidden_markov_states', '--n_hidden_markov_states', type = int, dest = 'n_hidden_markov_states', default = 2, help = 'number of hidden markov states to use')
  parser.add_argument('-n_latency_days', '--n_latency_days', type = int, dest = 'n_latency_days', default = 1000, help = 'number of latency days to use for hidden markov chain')
  parser.add_argument('-n_bins_hidden_var', '--n_bins_hidden_var', type = int, dest = 'n_bins_hidden_var', default = 2, help = 'number of bins to use to quantize primary hidden variable in hidden markov chain')
  parser.add_argument('-n_bins_hidden_var_secondary', '--n_bins_hidden_var_secondary', type = int, dest = 'n_bins_hidden_var_secondary', default = 2, help = 'number of bins to use to quantize secondary hidden variables')
  parser.add_argument('-n_hmm_predict_days', '--n_hmm_predict_days', type = int, dest = 'n_hmm_predict_days', default = 100, help = 'number of days to predict stock price using hmm')
  parser.add_argument('-test_set_averaged', '--test_set_averaged', type = int, dest = 'test_set_averaged', default = 1, help = 'set to one to average over the number of days in week for the test set, set to zero to not do this')
  parser.add_argument('-col_std', '--col_std', type = int, dest = 'col_std', default = 1, help = 'col_std used, recommended to be set to one')
  parser.add_argument('-col_mean', '--col_mean', type = int, dest = 'col_mean', default = 1, help = 'col_mean used, recommended to be set to one')
  args = parser.parse_args()


  #Organize and format input and output
  make_output_dir(args.output_dir)
  start_time = time.time()
 
  stock_objects_list = list() 
  stock_objects_names = list() 

  input_file = open(args.input_file, "r")

  available_files = []
  for file_name in input_file:
    if len(file_name.strip()) > 0:
      file_name = file_name.rstrip()
      print(file_name)
      if(os.stat(file_name).st_size > args.min_file_size): #ignore files with less than ~5 entries, as they are unlikely to be informative
        with open(file_name, 'r') as in_file:
          lines = in_file.read().splitlines()
          columns_line = lines[0]
          first_line = lines[1]
          last_line = lines[-1]
          if columns_line == args.anticipated_columns: #ignore files with missing columns
            first_line = first_line[:first_line.find(',')].split('-')
            last_line = last_line[:last_line.find(',')].split('-')
            first_line = [int(i) for i in first_line]
            last_line = [int(i) for i in last_line]

            if first_line[0] < int(args.min_year) or first_line[0] == int(args.min_year) and first_line[1] == int(args.min_month): #require first date to be 01/08 or earlier
              if last_line[0] > int(args.max_year) or last_line[0] == int(args.max_year) and last_line[1] == int(args.max_month): #require last date to be 10/16 or later
                available_files.append(file_name)
            else:
              print('{} does not have stock data from the minimum date range {}/{} - {}/{}'.format(file_name, args.min_month, args.min_year, args.max_month, args.max_year))
          else:
            print('{} is missing some -anticipated_columns, skipping.'.format(file_name))

  if args.number_files != -1:
    print('selecting {} random files from {} input files'.format(args.number_files, len(available_files)))
    available_files = stock_random.sample(available_files, args.number_files)
 
  for i in range(len(available_files)):
    file_name = available_files[i]
    file_name = file_name.rstrip()
    make_nested_dir(args.output_dir, get_stock_name(file_name))
    test_set_unscaled,train_set_unscaled,year_test_set_unscaled, all_data_set_unscaled, test_set, train_set, year_test_set, all_data_set, scaler = make_test_train_datasets(file_name, args)
    if type(test_set) != None and type(train_set) != None:
      stock_objects_list.append(stock_object_class(file_name, get_stock_name(file_name), test_set_unscaled, train_set_unscaled, year_test_set_unscaled, all_data_set_unscaled, test_set, train_set, year_test_set, all_data_set, scaler, args))
      stock_objects_names.append(get_stock_name(file_name))
      if i % 100 == 0:
        print('Datasets made for {} of {} files in {} seconds.'.format(i, len(available_files), time.time() - start_time))

  print('number of items actually selected: {}'.format(len(stock_objects_list)))

  #Plot Data
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
          plt.plot('Date', var, data = stock_objects_list[stock].train_set, markerfacecolor = c, linewidth = 0.4, label = stock_objects_list[stock].stock_name)
      #plt.legend()
      plt.xlabel('Date')
      plt.ylabel(var)
      plt.savefig(args.output_dir + '/stock_' + var + '_overlay.pdf')    
      plt.close('all')

  #Model Data
  for stock in stock_objects_list:
    if args.poly_reg == 1:
      print('DOING POLYNOMINAL FITS')
      degrees = [1, 2, 3, 4]
      for n in degrees:
        print('hi')
        poly_fit(stock, n)
    print('xgb predict')
    xgb_predict(stock)
    print('xgb sequential predict')


    n_estimators = [2, 3, 4]
    max_depth = [2, 3, 4, 5, 6]
    learning_rate = [0.05, 0.07, 0.1]
    min_child_weight = [0.01, 0.1, 1]
    subsample = [0.5, 1]

    for x in itertools.product(n_estimators, max_depth, learning_rate, min_child_weight, subsample):
      n = x[0]
      d = x[1]
      l = x[2]
      min_child = x[3]
      sub = x[4]
      #xgb_sequential_predict(stock, n, d, l, min_child, sub)
      #print('----------------------------')
      #print('n_estimators:{} max_depth:{} learning_rate: {} min_child_weight: {} subsample: {} '.format(n, d, l, min_child, sub))
      #print(stock.unscaled_model_names)
      #print(stock.unscaled_errors)
      overlay_predictions(stock)
    xgb_sequential_predict(stock, 4, 6, 0.1, 0.01, 0.5)

  print('----- {} seconds ---'.format(time.time() - start_time))
