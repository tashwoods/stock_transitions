def get_hmm_features(stock_object):
  Close_Open_Change = np.round(np.array(stock_object['Close_Open_Change']),1)
  High_Open_Change = np.round(np.array(stock_object['High_Open_Change']),1)
  Low_Open_Change = np.round(np.array(stock_object['Low_Open_Change']),1)

  feature_vector = np.column_stack((Close_Open_Change, High_Open_Change, Low_Open_Change))
  return feature_vector

def hmm_possible_outcomes(stock_object, n_steps, n_steps_secondary):
  frac_change_range = np.linspace(-2, 2, n_steps)
  frac_high_range = np.linspace(-2, 2, n_steps_secondary)
  frac_low_range = np.linspace(-2, 2, n_steps_secondary)
  possible_outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))
  return possible_outcomes

def get_most_probable_outcome_train_set(stock_object, train_set, test_set_array):
  #train hmm using train_set

  train_set_features = get_hmm_features(train_set)
  hmm = GaussianHMM(n_components = stock_object.input_args.n_hidden_markov_states, n_iter = 1000, verbose=True)
  hmm.monitor = ThresholdMonitor(hmm.monitor_.tol, hmm.monitor_.n_iter, hmm.monitor_.verbose)
  print('FITTING HMM ------------------------------------------')
  hmm.fit(train_set_features)
  print('DONE FITTING HMM --------------------------------------')
  print(hmm.monitor_.converged)

  most_probable_outcome = []
  predicted_open = []
  possible_outcomes = hmm_possible_outcomes(stock_object, stock_object.input_args.n_bins_hidden_var, stock_object.input_args.n_bins_hidden_var_secondary)
  
  #iterate over test_set array to obtain hmm prediction on test_set_array
  for test_set in test_set_array:
    test_set_features = get_hmm_features(test_set)
    outcome_score = []

    for possible_outcome in possible_outcomes: #calculate hmm score for each possible outcome
      total_data = np.row_stack((test_set_features, possible_outcome))
      outcome_score.append(hmm.score(total_data))


    frac_change, _, _ = possible_outcomes[np.argmax(outcome_score)]
    most_probable_outcome.append(frac_change)

    this_predicted_open = test_set.loc[0,stock_object.input_args.open_name]*(1 + frac_change)
    actual_open = test_set.loc[0,stock_object.input_args.open_name]
    error = 100*((predicted_open - actual_open)/actual_open)
    print('data: {} model: {} error: {}'.format(actual_open, predicted_open, error))

    predicted_open.append(this_predicted_open)

  return predicted_open


def hmm_get_close_prices_for_days(stock_object, dataset, start_index, days):
  predicted_close_prices = []
  for day_index in range(start_index, start_index + days):
    predicted_close_prices.append(get_close_price(stock_object, dataset, day_index))

  predicted_dates = dataset.iloc[start_index:start_index + days][stock_object.input_args.date_name]
  sampled_real_data = dataset.iloc[start_index:start_index + days][stock_object.input_args.close_price]
  #plt.plot(predicted_dates, stock_object.input_args.predict_var, data = stock_object.all_data_set, markerfacecolor = 'blue', linewidth = 0.4, label = 'All Dataset')
  plt.plot(predicted_dates, sampled_real_data, markerfacecolor = 'blue', label = 'Data')
  plt.plot(predicted_dates, predicted_close_prices, markerfacecolor = 'red', label = 'HMM')
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.xlim(-2, -1.5)
  plt.legend() 
  plt.savefig(stock_object.input_args.output_dir + '/stock_' + stock_object.stock_name + '_hmm_overlay.pdf')    
  plt.close('all')

def get_most_probable_outcome(stock_object, dataset, day_index):
  previous_data_start_index = max(0, day_index - stock_object.input_args.n_latency_days)
  previous_data_end_index = max(0, day_index - 1)
  print('start index: {} last index: {}'.format(previous_data_start_index, previous_data_end_index))
  previous_data = dataset.iloc[previous_data_start_index:previous_data_end_index]
  previous_data_features = get_hmm_features(previous_data)

  hmm = GaussianHMM(n_components = stock_object.input_args.n_hidden_markov_states)
  hmm.fit(previous_data_features)

  outcome_score = []
  possible_outcomes = hmm_possible_outcomes(stock_object, stock_object.input_args.n_bins_hidden_var, stock_object.input_args.n_bins_hidden_var_secondary)
  for possible_outcome in possible_outcomes:
    total_data = np.row_stack((previous_data_features, possible_outcome))
    outcome_score.append(hmm.score(total_data))
  most_probable_outcome = possible_outcomes[np.argmax(outcome_score)] 

  return most_probable_outcome

def hmm_get_close_prices_train_set(stock_object, train_set, test_set):
  weekly_split_data, weekly_dataframe = split_dataframe_into_dataframes(stock_object.year_test_set, stock_object.input_args.days_in_week)
  monthly_split_data, monthly_dataframe= split_dataframe_into_dataframes(stock_object.year_test_set, stock_object.input_args.days_in_month)
  predicted_close_prices = []

  predicted_close_prices = get_most_probable_outcome_train_set(stock_object, stock_object.train_set, weekly_split_data)

  #Plot Data and Predictions
  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.train_set, markerfacecolor = 'blue', linewidth = 0.4, label = 'Train Set')
  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.test_set, markerfacecolor = 'red', linewidth = 0.4, label = 'Test Set')
  plt.plot(weekly_dataframe[stock_object.input_args.date_name], predicted_close_prices, markerfacecolor = 'yellow', linewidth = 0.4, label = 'HMM')
  plt.xlim(1.5,1.9)
  plt.legend()
  plt.savefig(stock_object.input_args.output_dir + '/stock_' + stock_object.stock_name + '_hmm_overlay.pdf')    
  
def get_close_price(stock_object, dataset, day_index):
  open_price = dataset.iloc[day_index][stock_object.input_args.open_name]
  close_price = dataset.iloc[day_index][stock_object.input_args.close_name]
  predicted_change, _, _ = get_most_probable_outcome(stock_object, dataset, day_index)
  prediction = open_price * (1 + predicted_change)
  error = (abs(close_price - prediction)/close_price)*100
  print('open: {} close: {} prediction: {} error: {}'.format(open_price, close_price, prediction, error))
  return prediction

