from imported_libraries import *

def get_x_y_test_train(stock_object):
  predict_var = stock_object.input_args.predict_var
  date_name = stock_object.input_args.date_name

  x_train_set = stock_object.train_set.drop(predict_var, axis = 1)
  y_train_set = stock_object.train_set[[predict_var]]
  x_test_set = stock_object.test_set.drop(predict_var, axis = 1)
  y_test_set = stock_object.test_set[[predict_var]]

  return x_train_set, y_train_set, x_test_set, y_test_set

def get_unscaled_data(dataset, stock_object):
  array_data = stock_object.scaler.inverse_transform(dataset)
  formatted_data = pd.DataFrame(array_data, index = dataset.index, columns = dataset.columns)
  return formatted_data

def get_scaled_prediction_dataframe(x_test_set, test_prediction, stock_object):
  test_prediction_scaled = pd.DataFrame(test_prediction * stock_object.input_args.col_std + stock_object.input_args.col_mean)
  prediction_dataframe = x_test_set
  prediction_dataframe[stock_object.input_args.predict_var] = test_prediction_scaled.values
  return prediction_dataframe

def xgb_sequential_predict(stock_object):
  x_train_set, y_train_set, x_test_set, y_test_set = get_x_y_test_train(stock_object)
  test_prediction = []

  for i in range(len(x_test_set.index)):
    if i != 0:
      x_train_set = x_train_set.append(x_test_set.loc[i-1])
      y_train_set = y_train_set.append(y_test_set.loc[i-1])
    model = XGBRegressor(seed = 100, n_estimators = 5, max_depth = 5, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree = stock_object.input_args.col_std, colsample_bylevel = stock_object.input_args.col_mean, gamma = 0.1)
    model.fit(x_train_set.values, y_train_set.values)
    test_prediction.append(float(model.predict(x_test_set.iloc[i].values.reshape(1,-1))))
  
  test_prediction = np.asarray(test_prediction)
  prediction_dataframe = get_scaled_prediction_dataframe(x_test_set, test_prediction, stock_object)

  unscaled_train_set = get_unscaled_data(stock_object.train_set, stock_object)
  unscaled_test_set = get_unscaled_data(stock_object.test_set, stock_object)
  unscaled_prediction_set = get_unscaled_data(prediction_dataframe, stock_object)

  weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set.to_numpy().tolist())

  plt.plot(unscaled_test_set[stock_object.input_args.date_name], unscaled_test_set[stock_object.input_args.predict_var], label = 'Test set')
  plt.plot(unscaled_train_set[stock_object.input_args.date_name], unscaled_train_set[stock_object.input_args.predict_var], label = 'Train Set')
  plt.plot(unscaled_prediction_set[stock_object.input_args.date_name], unscaled_prediction_set[stock_object.input_args.predict_var], label = 'XGB Prediction RMSE = {}'.format(weekly_total_error))

  plt.legend()
  plt.xlim(2015,2018)
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': XGB Seq.')
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/' + 'stock_' + stock_object.stock_name + '_xgb_seq_overlay.pdf')    
  plt.close('all')

def xgb_predict(stock_object):
  x_train_set, y_train_set, x_test_set, y_test_set = get_x_y_test_train(stock_object)

  model = XGBRegressor(seed = 100, n_estimators = 5, max_depth = 5, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree = stock_object.input_args.col_std, colsample_bylevel = stock_object.input_args.col_mean, gamma = 0.1)
  model.fit(x_train_set, y_train_set)
  test_prediction = model.predict(x_test_set)
  print('test_predcition xgb predict')
  print(test_prediction)
  print(type(test_prediction))
  prediction_dataframe = get_scaled_prediction_dataframe(x_test_set, test_prediction, stock_object)

  unscaled_train_set = get_unscaled_data(stock_object.train_set, stock_object)
  unscaled_test_set = get_unscaled_data(stock_object.test_set, stock_object)
  unscaled_prediction_set = get_unscaled_data(prediction_dataframe, stock_object)

  weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set.to_numpy().tolist())

  plt.plot(unscaled_test_set[stock_object.input_args.date_name], unscaled_test_set[stock_object.input_args.predict_var], label = 'Test set')
  plt.plot(unscaled_train_set[stock_object.input_args.date_name], unscaled_train_set[stock_object.input_args.predict_var], label = 'Train Set')
  plt.plot(unscaled_prediction_set[stock_object.input_args.date_name], unscaled_prediction_set[stock_object.input_args.predict_var], label = 'XGB Prediction RMSE = {}'.format(weekly_total_error))

  plt.legend()
  plt.xlim(2015,2018)
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': XGB')
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/' + 'stock_' + stock_object.stock_name + '_xgb_overlay.pdf')    
  plt.close('all')

def poly_fit(stock_object, n):
  x_train_set, y_train_set, x_test_set, y_test_set = get_x_y_test_train(stock_object)
  model = np.poly1d(np.polyfit(stock_object.train_set[stock_object.input_args.date_name], stock_object.train_set[stock_object.input_args.predict_var], n))
  test_prediction = model(x_test_set[stock_object.input_args.date_name])
  weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set.to_numpy().tolist())
  print('Degree: {} Weekly error: {}'.format(n, weekly_total_error))

  unscaled_train_set = get_unscaled_data(stock_object.train_set, stock_object)
  unscaled_test_set = get_unscaled_data(stock_object.test_set, stock_object)

  prediction_dataframe = get_scaled_prediction_dataframe(x_test_set, test_prediction, stock_object)
  unscaled_prediction_set = get_unscaled_data(prediction_dataframe, stock_object)

  plt.plot(unscaled_test_set[stock_object.input_args.date_name], unscaled_test_set[stock_object.input_args.predict_var], label = 'Test set')
  plt.plot(unscaled_train_set[stock_object.input_args.date_name], unscaled_train_set[stock_object.input_args.predict_var], label = 'Train Set')
  plt.plot(unscaled_prediction_set[stock_object.input_args.date_name], unscaled_prediction_set[stock_object.input_args.predict_var], label = '{} Degree Fit RMSE = {}'.format(n, weekly_total_error))
  plt.legend()
  plt.xlim(2015,2018)

  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': ' + str(n) + ' Degree Polynomial Fit')
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/stock_' + stock_object.stock_name + '_poly'+ str(n) + '_overlay.pdf')    
  plt.close('all')

def get_general_errors_dataframes(model, data):
  error_array = model - data
  error_square_array = error_array**2
  total_error = round(math.sqrt((1/len(error_square_array))*sum(error_square_array)),2)
  print('total_error')
  print(total_error)
  return total_error

def split_dataframe_into_dataframes(dataframe, chunk_size):
  index_marks = range(0,len(dataframe.index), chunk_size)
  dataframes_array = []
  for i in index_marks:
    dataframes_array.append(dataframe[i:i+1].mean().to_frame().T)
  real_combined_dataframe = pd.concat(dataframes_array)
  return dataframes_array, real_combined_dataframe



