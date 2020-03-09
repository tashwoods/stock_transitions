from imported_libraries import *

def xgb_predict(stock_object):
  print('Beginning XGB Predict')
  col_std = 1
  col_mean  = 1
  predict_var = stock_object.input_args.predict_var
  date_name = stock_object.input_args.date_name

  x_train_set = stock_object.train_set.drop(predict_var, axis = 1)
  y_train_set = stock_object.train_set[[predict_var]]
  x_test_set = stock_object.test_set.drop(predict_var, axis = 1)
  y_test_set = stock_object.test_set[[predict_var]]

  model = XGBRegressor(seed = 100, n_estimators = 5, max_depth = 5, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree = col_std, colsample_bylevel = col_mean, gamma = 0.1)
  model.fit(x_train_set, y_train_set)
  test_prediction = model.predict(x_test_set)
  test_prediction_scaled = pd.DataFrame(test_prediction * col_std + col_mean)

  prediction_dataframe = x_test_set
  prediction_dataframe[predict_var] = test_prediction_scaled.values

  unscaled_train_set = get_unscaled_data(stock_object.train_set, stock_object)
  unscaled_test_set = get_unscaled_data(stock_object.test_set, stock_object)
  unscaled_prediction_set = get_unscaled_data(prediction_dataframe, stock_object)

  print('actual')
  print(unscaled_test_set)
  print('predicted')
  print(unscaled_prediction_set[predict_var])

  plt.plot(unscaled_train_set[date_name], unscaled_train_set[predict_var], label = 'Train Set')
  plt.plot(unscaled_test_set[date_name], unscaled_test_set[predict_var], label = 'Test set')
  plt.plot(unscaled_prediction_set[date_name], unscaled_prediction_set[predict_var], label = 'XGB Prediction')
  plt.legend()
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/' + 'stock_' + stock_object.stock_name + '_xgb_overlay.pdf')    

def get_unscaled_data(dataset, stock_object):
  array_data = stock_object.scaler.inverse_transform(dataset)
  formatted_data = pd.DataFrame(array_data, index = dataset.index, columns = dataset.columns)

  return formatted_data

def poly_fit(stock_object, n):
  model = np.poly1d(np.polyfit(stock_object.train_set[stock_object.input_args.date_name], stock_object.train_set[stock_object.input_args.predict_var], n))

  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.train_set, markerfacecolor = 'blue', linewidth = 0.4, label = 'Train Set')
  plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.test_set, markerfacecolor = 'red', linewidth = 0.4, label = 'Test Set')
  #plt.plot(stock_object.input_args.date_name, stock_object.input_args.predict_var, data = stock_object.year_test_set, markerfacecolor = 'yello', linewidth = 0.4, label = 'Year Test Set')

  date_min = stock_object.all_data_set[stock_object.input_args.date_name].min()
  date_max = stock_object.all_data_set[stock_object.input_args.date_name].max()
  x = np.linspace(date_min, date_max)
  y = np.arange(0,10)
  plt.plot(x,model(x), '--', label = 'Model')
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)

  #weekly_split_data = averaged_dataframe_array(stock_object.year_test_set, stock_object.input_args.days_in_week)
  #monthly_split_data = averaged_dataframe_array(stock_object.year_test_set, stock_object.input_args.days_in_month)
  #weekly_total_error, weekly_error_array, weekly_date_array, weekly_data_array = get_general_errors(stock_object, weekly_split_data, model)
  #monthly_total_error, monthly_error_array, monthly_date_array, monthly_data_array = get_general_errors(stock_object, monthly_split_data, model)
  #print('Degree: {} Weekly error: {} Monthly error: {}'.format(n, weekly_total_error, monthly_total_error))

  weekly_total_error, weekly_error_array, weekly_date_array, weekly_data_array = get_general_errors_dataframes(stock_object, stock_object.test_set, model)
  print('Degree: {} Weekly error: {}'.format(n, weekly_total_error))
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': ' + str(n) + ' Degree Polynomial Fit')

  #plt.plot(monthly_date_array, monthly_data_array, '-c', label = 'Averaged Monthly, RMSE = {}'.format(monthly_total_error))
  plt.plot(weekly_date_array, weekly_data_array, '-y', label = 'Averaged Weekly, RMSE = {}'.format(weekly_total_error))

  plt.ylim(1.2*min(stock_object.all_data_set[stock_object.input_args.predict_var]), 1.2*max(stock_object.all_data_set[stock_object.input_args.predict_var]))
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/stock_' + stock_object.stock_name + '_poly'+ str(n) + '_overlay.pdf')    
  plt.close('all')

def get_general_errors_dataframes(stock_object, input_data, model):
  error_array = []
  error_square_array = []
  date_array = []
  data_array = []
  date_column_index = input_data.columns.get_loc(stock_object.input_args.date_name)
  predict_var_column_index = input_data.columns.get_loc(stock_object.input_args.predict_var)
  for i in range(len(input_data.index)):
    date = input_data.iloc[i, date_column_index]
    data = input_data.iloc[i, predict_var_column_index]
    predic = model(date)
    date_array.append(date)
    error_array.append(predic-data)
    error_square_array.append((predic - data)**2)
    data_array.append(data)

  total_error = round(math.sqrt((1/len(error_square_array))*sum(error_square_array)),2)
  print('date')
  print(date_array)
  print('new error array')
  print(error_array)
  print('total error')
  print(total_error)
  
  return total_error, error_array, date_array, data_array

def split_dataframe_into_dataframes(dataframe, chunk_size):
  index_marks = range(0,len(dataframe.index), chunk_size)
  dataframes_array = []
  for i in index_marks:
    dataframes_array.append(dataframe[i:i+1].mean().to_frame().T)
  real_combined_dataframe = pd.concat(dataframes_array)
  return dataframes_array, real_combined_dataframe



