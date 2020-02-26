from imported_libraries import *

def linear_predict_stocks(stock_object):
  date = stock_object.train_set[stock_object.input_args.date_name]
  open_price = stock_object.train_set[stock_object.input_args.predict_var]
  
  open_price = sm.add_constant(open_price) #add intercept to model 
  model = sm.OLS(date, open_price).fit()
  predictions = model.predict(open_price) 
  const, slope = model.params

  print(model.summary())

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

  
  weekly_split_data = averaged_dataframe(stock_object, stock_object.input_args.days_in_week)
  monthly_split_data = averaged_dataframe(stock_object, stock_object.input_args.days_in_month)

  weekly_total_error, weekly_error_array, weekly_date_array, weekly_data_array = get_errors(stock_object, weekly_split_data, model)
  monthly_total_error, monthly_error_array, monthly_date_array, monthly_data_array = get_errors(stock_object, monthly_split_data, model)
  
  print('weekly error: {} monthly error: {}'.format(weekly_total_error, monthly_total_error))

  plt.plot(monthly_date_array, monthly_data_array, '-c', label = 'Averaged Monthly, RMSE = {}'.format(monthly_total_error))
  plt.plot(weekly_date_array, weekly_data_array, '-y', label = 'Averaged Weekly, RMSE = {}'.format(weekly_total_error))
  plt.xlim(1.5,1.7)
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/stock_' + stock_object.stock_name + '_linreg_overlay.pdf')    
  plt.close('all')

def poly_fit(stock_object, n):
  model = np.poly1d(np.polyfit(stock_object.train_set[stock_object.input_args.date_name], stock_object.train_set[stock_object.input_args.predict_var], n))

  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.train_set, markerfacecolor = 'blue', linewidth = 0.4, label = 'Train Set')
  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.test_set, markerfacecolor = 'red', linewidth = 0.4, label = 'Test Set')
  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.year_test_set, markerfacecolor = 'yello', linewidth = 0.4, label = 'Year Test Set')

  date_min = stock_object.all_data_set[stock_object.input_args.date_name].min()
  date_max = stock_object.all_data_set[stock_object.input_args.date_name].max()
  x = np.linspace(date_min, date_max)
  x = np.linspace(1,2)
  y = np.arange(0,10)
  plt.plot(x,model(x), '--', label = 'Model')
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)

  
  weekly_split_data = averaged_dataframe(stock_object, stock_object.input_args.days_in_week)
  monthly_split_data = averaged_dataframe(stock_object, stock_object.input_args.days_in_month)

  weekly_total_error, weekly_error_array, weekly_date_array, weekly_data_array = get_general_errors(stock_object, weekly_split_data, model)
  monthly_total_error, monthly_error_array, monthly_date_array, monthly_data_array = get_general_errors(stock_object, monthly_split_data, model)
  
  print('Degree: {} Weekly error: {} Monthly error: {}'.format(n, weekly_total_error, monthly_total_error))
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': ' + str(n) + ' Degree Polynomial Fit')

  plt.plot(monthly_date_array, monthly_data_array, '-c', label = 'Averaged Monthly, RMSE = {}'.format(monthly_total_error))
  plt.plot(weekly_date_array, weekly_data_array, '-y', label = 'Averaged Weekly, RMSE = {}'.format(weekly_total_error))


  plt.xlim(1.5,1.7)
  plt.ylim(1.2*min(stock_object.all_data_set[stock_object.input_args.date_name]), 1.2*max(stock_object.all_data_set[stock_object.input_args.predict_var]))
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/stock_' + stock_object.stock_name + '_poly'+ str(n) + '_overlay.pdf')    
  plt.close('all')
 
def get_general_errors(stock_object, input_data, model):
  error_array = []
  error_square_array = []
  date_array = []
  data_array = []
  for i in input_data:
    data = i[stock_object.input_args.predict_var].mean()
    date = i[stock_object.input_args.date_name].mean()
    predic = model(date)
    date_array.append(date)
    error_array.append(predic-data)
    error_square_array.append((predic - data)**2)
    data_array.append(data)

  total_error = round(math.sqrt((1/len(error_square_array))*sum(error_square_array)),2)
  
  return total_error, error_array, date_array, data_array



def get_errors(stock_object, input_data, model):
  slope, const = model.params
  error_array = []
  error_square_array = []
  date_array = []
  data_array = []
  for i in input_data:
    data = i[stock_object.input_args.predict_var].mean()
    date = i[stock_object.input_args.date_name].mean()
    predic = slope*date + const
    date_array.append(date)
    error_array.append(predic-data)
    error_square_array.append((predic - data)**2)
    data_array.append(data)

  total_error = round(math.sqrt((1/len(error_square_array))*sum(error_square_array)),2)
  
  return total_error, error_array, date_array, data_array

def averaged_dataframe(stock_object, days):
  split_data = [stock_object.year_test_set[i:i+days] for i in range(0,stock_object.year_test_set.shape[0],days)]
  return split_data

def get_hmm_features(stock_object):
  Close_Open_Change = np.array(stock_object['Close_Open_Change']) 
  High_Open_Change = np.array(stock_object['High_Open_Change'])
  Low_Open_Change = np.array(stock_object['Low_Open_Change'])

  feature_vector = np.column_stack((Close_Open_Change, High_Open_Change, Low_Open_Change))
  return feature_vector

def hmm_prediction(stock_object):
  get_most_probable_outcome(stock_object, 50)
  return 

def get_most_probable_outcome(stock_object, day_index):

  previous_data_start_index = max(0, day_index - stock_object.input_args.n_latency_days)
  previous_data_end_index = max(0, day_index - 1)
  previous_data = stock_object.train_set.iloc[previous_data_start_index:previous_data_end_index]
  previous_data_features = get_hmm_features(previous_data)

  hmm = GaussianHMM(n_components = stock_object.input_args.n_hidden_markov_states)
  hmm.fit(previous_data_features)

  outcome_score = []
  possible_outcomes = hmm_possible_outcomes(stock_object, stock_object.input_args.n_bins_hidden_var)
  for possible_outcome in possible_outcomes:
    total_data = np.row_stack((previous_data_features, possible_outcome))
    outcome_score.append(hmm.score(total_data))
  most_probable_outcome = possible_outcomes[np.argmax(outcome_score)] 

  print('most probable outcome')
  print(most_probable_outcome)
  return previous_data_features

def hmm_possible_outcomes(stock_object, n_steps):
  frac_change_range = np.linspace(-0.1, 0.1, n_steps)
  frac_high_range = np.linspace(-0.1, 0.1, n_steps)
  frac_low_range = np.linspace(-0.1, 0.1, n_steps)
  possible_outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))
  return possible_outcomes
