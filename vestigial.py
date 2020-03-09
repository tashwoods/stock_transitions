#Extra plots
#for var in stock_object.train_set.columns:
  #stock_object.make_histograms(var)
#stock_object.make_scatter_plots() #scatter plots without z axis coloring by third variable

def linear_predict_stocks(stock_object):
  date = stock_object.train_set[stock_object.input_args.date_name]
  open_price = stock_object.train_set[stock_object.input_args.predict_var]
  
  open_price = sm.add_constant(open_price) #add intercept to model 
  model = sm.OLS(date, open_price).fit()
  predictions = model.predict(open_price) 
  const, slope = model.params

  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.train_set, markerfacecolor = 'blue', linewidth = 0.4, label = 'Train Set')
  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.test_set, markerfacecolor = 'red', linewidth = 0.4, label = 'Test Set')
  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.year_test_set, markerfacecolor = 'yello', linewidth = 0.4, label = 'Year Test Set')

  date_min = stock_object.train_set[stock_object.input_args.date_name].min()
  date_max = stock_object.train_set[stock_object.input_args.date_name].max()
  x = np.linspace(date_min, date_max)
  x = np.linspace(1,2)
  y = np.arange(0,10)
  plt.plot(x, slope*x + const, '-', label = 'Model')
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)

  
  weekly_split_data = averaged_dataframe_array(stock_object.year_test_set, stock_object.input_args.days_in_week)
  monthly_split_data = averaged_dataframe_array(stock_object.year_test_set, stock_object.input_args.days_in_month)

  weekly_total_error, weekly_error_array, weekly_date_array, weekly_data_array = get_errors(stock_object, weekly_split_data, model)
  monthly_total_error, monthly_error_array, monthly_date_array, monthly_data_array = get_errors(stock_object, monthly_split_data, model)
  
  print('weekly error: {} monthly error: {}'.format(weekly_total_error, monthly_total_error))

  plt.plot(monthly_date_array, monthly_data_array, '-c', label = 'Averaged Monthly, RMSE = {}'.format(monthly_total_error))
  plt.plot(weekly_date_array, weekly_data_array, '-y', label = 'Averaged Weekly, RMSE = {}'.format(weekly_total_error))
  plt.xlim(1.5,1.7)
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/stock_' + stock_object.stock_name + '_linreg_overlay.pdf')    
  plt.close('all')
