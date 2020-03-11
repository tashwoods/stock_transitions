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

    model = XGBRegressor(n_estimators = 5, max_depth = 5, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree = stock_object.input_args.col_std, colsample_bylevel = stock_object.input_args.col_mean, gamma = 0.1)
    model.fit(x_train_set.values, y_train_set.values)
    test_prediction.append(float(model.predict(x_test_set.iloc[i].values.reshape(1,-1))))
  
  test_prediction = np.asarray(test_prediction)
  scaled_prediction_set = get_scaled_prediction_dataframe(x_test_set, test_prediction, stock_object)
  unscaled_prediction_set = get_unscaled_data(scaled_prediction_set, stock_object)

  scaled_weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set.to_numpy().tolist())
  unscaled_weekly_total_error = get_general_errors_dataframes(unscaled_prediction_set[stock_object.input_args.predict_var], stock_object.test_set_unscaled[stock_object.input_args.predict_var])
  print('UnScaled XGB error: {}'.format(unscaled_weekly_total_error))
  print('Scaled XGB error: {}'.format(scaled_weekly_total_error))

  prediction_overlay_plot(stock_object.test_set, stock_object.train_set, scaled_prediction_set, scaled_weekly_total_error, 'XGBScaled', stock_object)
  prediction_overlay_plot(stock_object.test_set_unscaled, stock_object.train_set_unscaled, unscaled_prediction_set, unscaled_weekly_total_error, 'XGBUnScaled', stock_object)

  stock_object.add_unscaled_model("SeqXGBUnScaled", unscaled_prediction_set, unscaled_weekly_total_error)
  stock_object.add_scaled_model("SeqXGBScaled", scaled_prediction_set, scaled_weekly_total_error)
  
def xgb_predict(stock_object):
  x_train_set, y_train_set, x_test_set, y_test_set = get_x_y_test_train(stock_object)
  model = XGBRegressor(seed = 100, n_estimators = 5, max_depth = 5, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree = stock_object.input_args.col_std, colsample_bylevel = stock_object.input_args.col_mean, gamma = 0.1)
  model.fit(x_train_set, y_train_set)

  test_prediction = model.predict(x_test_set)
  scaled_prediction_set = get_scaled_prediction_dataframe(x_test_set, test_prediction, stock_object)
  unscaled_prediction_set = get_unscaled_data(scaled_prediction_set, stock_object)

  scaled_weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set.to_numpy().tolist())
  unscaled_weekly_total_error = get_general_errors_dataframes(unscaled_prediction_set[stock_object.input_args.predict_var], stock_object.test_set_unscaled[stock_object.input_args.predict_var])
  print('UnScaled XGB error: {}'.format(unscaled_weekly_total_error))
  print('Scaled XGB error: {}'.format(scaled_weekly_total_error))

  prediction_overlay_plot(stock_object.test_set, stock_object.train_set, scaled_prediction_set, scaled_weekly_total_error, 'XGBScaled', stock_object)
  prediction_overlay_plot(stock_object.test_set_unscaled, stock_object.train_set_unscaled, unscaled_prediction_set, unscaled_weekly_total_error, 'XGBUnScaled', stock_object)
  stock_object.add_unscaled_model("XGBUnScaled", unscaled_prediction_set, unscaled_weekly_total_error)
  stock_object.add_scaled_model("XGBScaled", scaled_prediction_set, scaled_weekly_total_error)

def poly_fit(stock_object, n):
  model = np.poly1d(np.polyfit(stock_object.train_set[stock_object.input_args.date_name], stock_object.train_set[stock_object.input_args.predict_var], n))

  x_train_set, y_train_set, x_test_set, y_test_set = get_x_y_test_train(stock_object)
  test_prediction = model(x_test_set[stock_object.input_args.date_name])
  scaled_prediction_set = get_scaled_prediction_dataframe(x_test_set, test_prediction, stock_object)
  unscaled_prediction_set = get_unscaled_data(scaled_prediction_set, stock_object)

  scaled_weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set.to_numpy().tolist())
  unscaled_weekly_total_error = get_general_errors_dataframes(unscaled_prediction_set[stock_object.input_args.predict_var], stock_object.test_set_unscaled[stock_object.input_args.predict_var])
  print('SCALED: Degree: {} Weekly error: {}'.format(n, scaled_weekly_total_error))
  print('UNSCALED: Degree: {} Weekly error: {}'.format(n, unscaled_weekly_total_error))

  prediction_overlay_plot(stock_object.test_set_unscaled, stock_object.train_set_unscaled, unscaled_prediction_set, unscaled_weekly_total_error, str(n) + 'PolyUnscaled', stock_object) 
  prediction_overlay_plot(stock_object.test_set, stock_object.train_set, scaled_prediction_set, scaled_weekly_total_error, str(n) + 'PolyScaled', stock_object)

  stock_object.add_unscaled_model('Poly' + str(n) + 'UnScaled' , unscaled_prediction_set, unscaled_weekly_total_error)
  stock_object.add_scaled_model('Poly' + str(n) + 'Scaled', scaled_prediction_set, scaled_weekly_total_error)

def overlay_predictions(stock_object):
  #Overlay Unscaled Predictions, Test, and Train Sets
  for name, model, rmse in zip(stock_object.unscaled_model_names, stock_object.unscaled_models, stock_object.unscaled_errors):
    print('hi')
    plt.plot(model[stock_object.input_args.date_name], model[stock_object.input_args.predict_var], label = '{} Fit RMSE = {}'.format(name, rmse))
  plt.plot(stock_object.test_set_unscaled[stock_object.input_args.date_name], stock_object.test_set_unscaled[stock_object.input_args.predict_var], label = 'Test set')
  plt.plot(stock_object.train_set_unscaled[stock_object.input_args.date_name], stock_object.train_set_unscaled[stock_object.input_args.predict_var], label = 'Train Set')

  plt.legend()
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': All UnScaled Models')
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/stock_models_all_unscaled_' + stock_object.stock_name + '_overlay.pdf')    
  plt.close('all')

  #Overlay Scaled Predictions, Test, and Train Sets
  for name, model, rmse in zip(stock_object.scaled_model_names, stock_object.scaled_models, stock_object.scaled_errors):
    print('hi')
    plt.plot(model[stock_object.input_args.date_name], model[stock_object.input_args.predict_var], label = '{} Fit RMSE = {}'.format(name, rmse))
  plt.plot(stock_object.test_set[stock_object.input_args.date_name], stock_object.test_set[stock_object.input_args.predict_var], label = 'Test set')
  plt.plot(stock_object.train_set[stock_object.input_args.date_name], stock_object.train_set[stock_object.input_args.predict_var], label = 'Train Set')

  plt.legend()
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': All Scaled Models')
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/stock_models_all_scaled_' + stock_object.stock_name + '_overlay.pdf')    
  plt.close('all')


def prediction_overlay_plot(test_set, train_set, model, rmse, name, stock_object):
  plt.plot(test_set[stock_object.input_args.date_name], test_set[stock_object.input_args.predict_var], label = 'Test set')
  plt.plot(train_set[stock_object.input_args.date_name], train_set[stock_object.input_args.predict_var], label = 'Train Set')
  plt.plot(model[stock_object.input_args.date_name], model[stock_object.input_args.predict_var], label = '{} Fit RMSE = {}'.format(name, rmse))

  plt.legend()
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': ' + name + ' Fit')
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/stock_model_' + name + '_' + stock_object.stock_name + '_overlay.pdf')    
  plt.close('all')

def get_general_errors_dataframes(model, data):
  error_array = []
  error_square_array = []
  for i,j in zip(model, data):
    error_array.append(abs(i - j))
    error_square_array.append(abs(i-j)**2)
  total_error = round(math.sqrt((1/len(error_square_array))*sum(error_square_array)),2)
  return total_error

def split_dataframe_into_dataframes(dataframe, chunk_size):
  index_marks = range(0,len(dataframe.index), chunk_size)
  dataframes_array = []
  for i in index_marks:
    dataframes_array.append(dataframe[i:i+1].mean().to_frame().T)
  real_combined_dataframe = pd.concat(dataframes_array)
  return dataframes_array, real_combined_dataframe



