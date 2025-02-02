from imported_libraries import *

def worker_plots(stock_object):
  #stock_object.make_overlay_plot() #overlay stock open, high, low, close vs. t plots
  #stock_object.make_scatter_heat_plots() #scatter plots with z axis coloring by third variable
  #stock_object.make_correlation_plots()
  #stock_object.make_time_dependent_plots() #for each stock variable individually
  return

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
  parser.add_argument('-scale_features', '--scale_features', type = int, dest = 'scale_features', default = 1, help = 'set to one to scale features using StandardScaler(), 0 to not, using entire dataset')
  parser.add_argument('-iteratively_scale_features', '--iteratively_scale_features', type = int, dest = 'iteratively_scale_features', default = 0, help = 'set to one to iteratively scale data points to train set plus all previous test set points')
  parser.add_argument('-combined_features', '--combined_features', type = int, dest = 'combined_features', default = 0, help = 'set to one to add combined features to dataset, zero to not')
  #parser.add_argument('-anticipated_columns', '--anticipated_columns', type = str, dest = 'anticipated_columns', default = 'Date,Open,High,Low,Close,Volume,OpenInt', help = 'list of columns that are expected in text files')
  parser.add_argument('-anticipated_columns', '--anticipated_columns', type = str, dest = 'anticipated_columns', default = 'Date,Open,High,Low,Close,Adj Close,Volume', help = 'list of columns that are expected in text files')
  parser.add_argument('-poly_reg', '--poly_reg', type = int, dest = 'poly_reg', default = 0, help = 'set to one to model stock open price with polynominal regression')
  parser.add_argument('-min_year', '--min_year', type = str, dest = 'min_year', default = '2018', help = 'first year to require stock data for')
  parser.add_argument('-max_year', '--max_year', type = str, dest = 'max_year', default = '2020', help = 'last year to require stock data for')
  parser.add_argument('-min_month', '--min_month', type = str, dest = 'min_month', default = '01', help = 'month of the first year to require stock data for')
  parser.add_argument('-max_month', '--max_month', type = str, dest = 'max_month', default = '01', help = 'month of the last year to require stock data for')
  parser.add_argument('-year_test_set', '--year_test_set', type = str, dest = 'year_test_set', default = '2020', help = 'year to begin test set with')
  parser.add_argument('-month_test_set', '--month_test_set', type = str, dest = 'month_test_set', default = '01', help = 'month to begin test set with')
  parser.add_argument('-day_test_set', '--day_test_set', type = str, dest = 'day_test_set', default = '01', help = 'day of month to begin test set with')
  parser.add_argument('-days_in_week', '--days_in_week', type = int, dest = 'days_in_week', default = 1, help = 'days in week to use for weekly averging')
  parser.add_argument('-days_in_month', '--days_in_month', type = int, dest = 'days_in_month', default = 30, help = 'days in month to use for month averging')
  parser.add_argument('-n_hidden_markov_states', '--n_hidden_markov_states', type = int, dest = 'n_hidden_markov_states', default = 2, help = 'number of hidden markov states to use')
  parser.add_argument('-n_latency_days', '--n_latency_days', type = int, dest = 'n_latency_days', default = 1000, help = 'number of latency days to use for hidden markov chain')
  parser.add_argument('-n_bins_hidden_var', '--n_bins_hidden_var', type = int, dest = 'n_bins_hidden_var', default = 2, help = 'number of bins to use to quantize primary hidden variable in hidden markov chain')
  parser.add_argument('-n_bins_hidden_var_secondary', '--n_bins_hidden_var_secondary', type = int, dest = 'n_bins_hidden_var_secondary', default = 2, help = 'number of bins to use to quantize secondary hidden variables')
  parser.add_argument('-n_hmm_predict_days', '--n_hmm_predict_days', type = int, dest = 'n_hmm_predict_days', default = 100, help = 'number of days to predict stock price using hmm')
  parser.add_argument('-test_set_averaged', '--test_set_averaged', type = int, dest = 'test_set_averaged', default = 1, help = 'set to one to average over the number of days in week for the test set, set to zero to not do this')
  args = parser.parse_args()

  #Organize and format input and output
  make_output_dir(args.output_dir)
  start_time = time.time()
  stock_objects_list = list() 
  stock_objects_names = list() 
  available_files = []

  #Process input file list
  input_file = open(args.input_file, "r")
  for file_name in input_file:
    if len(file_name.strip()) > 0:
      file_name = file_name.rstrip()
      if(os.stat(file_name).st_size > args.min_file_size): #ignore files smaller than min_file_size
        with open(file_name, 'r') as in_file: #iterate over remaining files
          lines = in_file.read().splitlines()
          #Check if date range file meets user specified date range and variable names
          columns_line = lines[0]
          first_line = lines[1]
          last_line = lines[-1]
          print('lines')
          print(first_line)
          print(last_line)
          if columns_line == args.anticipated_columns: #ignore files missing columns
            first_line = first_line[:first_line.find(',')].split('-')
            last_line = last_line[:last_line.find(',')].split('-')
            first_line = [int(i) for i in first_line]
            last_line = [int(i) for i in last_line]
            if first_line[0] < int(args.min_year) or first_line[0] == int(args.min_year) and first_line[1] <= int(args.min_month):
              if last_line[0] > int(args.max_year) or last_line[0] == int(args.max_year) and last_line[1] >= int(args.max_month):
                available_files.append(file_name)
            else:
              print('{} fails date range rqmt {}/{}-{}/{}'.format(file_name, args.min_month, args.min_year, args.max_month, args.max_year))
          else:
            print('{} missing some -anticipated_columns, skipping.'.format(file_name))

  print(available_files)
  if args.number_files != -1: #if number_files != -1 randomly select number_files specified
    print('selecting {} random files from {} input files'.format(args.number_files, len(available_files)))
    available_files = stock_random.sample(available_files, args.number_files)

  #For each valid file create output dir and stock_object and create meta-lists of these
  for i in range(len(available_files)):
    file_name = available_files[i]
    file_name = file_name.rstrip()
    make_nested_dir(args.output_dir, get_stock_name(file_name))
    test_set_unscaled, train_set_unscaled, all_data_set_unscaled = make_test_train_datasets(file_name, args)
    if i == 0:
      print('test_set')
      print(test_set_unscaled)
    if type(test_set_unscaled) != None and type(train_set_unscaled) != None:
      stock_objects_list.append(stock_object_class(file_name, get_stock_name(file_name), test_set_unscaled, train_set_unscaled, all_data_set_unscaled, args))
      stock_objects_names.append(get_stock_name(file_name))
      if i % 100 == 0:
        print('Datasets made for {} of {} files in {} seconds.'.format(i, len(available_files), time.time() - start_time))
  print('number of items selected: {}'.format(len(stock_objects_list)))

  #Plot Data
  if args.indiv_plots == 1:
    #Create multiple threads for speedy plotting
    pool = multiprocessing.Pool(args.max_number_processes)
    result_list = pool.map_async(worker_plots, stock_objects_list)
    pool.close()
    pool.join()

  if args.overlay_stock_plots == 1: #overlay time-dependent variable plots for all stocks
    for var in stock_objects_list[0].train_set_unscaled.columns: #iterate over stock variables
      color = cm.rainbow(np.linspace(0,1,len(stock_objects_list)))
      for stock,c in zip(range(len(stock_objects_list)), color): #iterate over stocks
        if var != args.date_name:
          plt.plot('Date', var, data = stock_objects_list[stock].train_set_unscaled, markerfacecolor = c, linewidth = 0.4, label = stock_objects_list[stock].stock_name)
      plt.xlabel('Date')
      plt.ylabel(var)
      plt.savefig(args.output_dir + '/stock_' + var + '_overlay.pdf')    
      plt.close('all')

  #Model Data
  for stock in stock_objects_list:
    if args.poly_reg == 1:
      degrees = [2]
      for n in degrees:
        poly_fit(stock, n)
        
    print('Simple XGB Fit')

    n_estimators = [950] #more is better
    max_depth = [90] #for estimators = 10, 20 was best
    learning_rate = [.13] #0.9 looked best
    min_child_weight = [.28] #seemed to have no effect
    subsample = [0.88]#1 seemed best
    xgb_predict(stock,950,90,0.13,0.28,0.88)

    def xgb_rmse(max_depth, n_estimators, learning_rate, min_child_weight, subsample):
      params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth, 'min_child_weight': min_child_weight, 'subsample': subsample}
      xgb_val = xgb_sequential_predict(stock, int(max_depth), int(n_estimators), learning_rate, min_child_weight, subsample)
      return -1.0 * xgb_val

    #xgb_bo = BayesianOptimization(xgb_rmse, {'n_estimators': (5,1000), 'max_depth': (3,100), 'learning_rate': (0.1,1), 'min_child_weight': (0,1), 'subsample': (0,1)})
    #xgb_bo.maximize(n_iter = 20, init_points = 20)
    #params = xgb_bo.max['params']
    #print('Simple XGB Best params')
    #print(params['max_depth'])
    #print(params))
    #xgb_predict(stock, params['n_estimators'],['max_depth'], params['learning_rate'], params['min_child_weight'], params['subsample'])

    print('XGB Sequential Fit')
    #xgb_bo = BayesianOptimization(xgb_rmse, {'n_estimators': (5,1000), 'max_depth': (3,100), 'learning_rate': (0.1,1), 'min_child_weight': (0,1), 'subsample': (0,1)})
    #xgb_bo.maximize(n_iter = 20, init_points = 20)
    #params = xgb_bo.max['params']
    #print('Seq XGB Best params')
    #print(params)
    #xgb_predict(stock, int(params['n_estimators']), int(params['max_depth']), params['learning_rate'], params['min_child_weight'], params['subsample'])

    scaled_rmse = []
    unscaled_rmse = []
    hyperparameter_array = []
    for x in itertools.product(n_estimators, max_depth, learning_rate, min_child_weight, subsample):
      print('----------------------------')
      print(x)
      print(x[0])
      unscaled_weekly_error = xgb_sequential_predict(stock,x[0], x[1], x[2], x[3], x[4]) 
      scaled_weekly_error = 0
      scaled_rmse.append(scaled_weekly_error)
      hyperparameter_array.append(x)
      unscaled_rmse.append(unscaled_weekly_error)
      print('n_estimators:{} max_depth:{} learning_rate: {} min_child_weight: {} subsample: {} '.format(x[0], x[1], x[2], x[3], x[4]))
      overlay_predictions(stock)

  print(hyperparameter_array)
  print(list(hyperparameter_array[0]))
  print(unscaled_rmse)

  print('Best hyperparameters:')
  min_index = unscaled_rmse.index(min(unscaled_rmse))
  print(unscaled_rmse[min_index])
  print(hyperparameter_array[min_index])
  x = [i[0] for i in hyperparameter_array]
  y = [i[1] for i in hyperparameter_array]
  plt.plot(x,unscaled_rmse)
  plt.xlabel('n_estimators')
  plt.ylabel('RMSE')
  plt.savefig('rmse_nestimators.pdf')
  data = pd.DataFrame({'x': x, 'y': y, 'z':unscaled_rmse})
  ax = sns.heatmap(data.pivot_table(index = 'y', columns = 'x', values = 'z'), annot = True, linewidths = 0.5)
  ax.invert_yaxis()
  ax.collections[0].colorbar.set_label('RMSE')
  plt.xlabel('n_estimators')
  plt.ylabel('max_depth')
  plt.savefig('rmse_hyperparameter.pdf')


  print('----- {} seconds ---'.format(time.time() - start_time))
