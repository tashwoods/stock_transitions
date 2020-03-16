from imported_libraries import *

def get_x_y_test_train(stock_object, scale, i):
  predict_var = stock_object.input_args.predict_var
  date_name = stock_object.input_args.date_name
  scaler = StandardScaler()

  if scale == -1:
    train_set = stock_object.train_set_unscaled
    test_set = stock_object.test_set_unscaled

  if scale == 0:#use scaled data from organize_data.py
    train_set = stock_object.train_set
    test_set = stock_object.test_set

  if scale == 1: #iteratively scale
    print(i)
    train_set = stock_object.train_set_unscaled
    test_set = stock_object.test_set_unscaled

    if i != 0:
      if i == 1:
        array_to_add = test_set.iloc[0]
        df_to_add = pd.DataFrame(array_to_add, index = test_set.index, columns = test_set.columns)
        train_set = train_set.append(df_to_add)
        test_set = test_set.iloc[i:]
      else:
        train_set = train_set.append(test_set.iloc[:i-1])
        test_set = test_set.iloc[i:]

    train_set_features = scaler.fit_transform(train_set)
    test_set_features = scaler.transform(test_set)
    final_train_set = pd.DataFrame(train_set_features, index = train_set.index, columns = train_set.columns)
    final_test_set = pd.DataFrame(test_set_features, index = test_set.index, columns = test_set.columns)
    train_set = final_train_set
    test_set = final_test_set

  x_train_set = train_set.drop(predict_var, axis = 1)
  y_train_set = train_set[[predict_var]]
  x_test_set = test_set.drop(predict_var, axis = 1)
  y_test_set = test_set[[predict_var]]

  return x_train_set, y_train_set, x_test_set, y_test_set, scaler

def get_unscaled_data(dataset, stock_object):
  array_data = stock_object.scaler.inverse_transform(dataset)
  formatted_data = pd.DataFrame(array_data, index = dataset.index, columns = dataset.columns)
  return formatted_data

def get_prediction_dataframe(x_test_set, test_prediction, stock_object):
  test_prediction_scaled = pd.DataFrame(test_prediction * stock_object.input_args.col_std + stock_object.input_args.col_mean)
  prediction_dataframe = x_test_set
  prediction_dataframe[stock_object.input_args.predict_var] = test_prediction_scaled.values
  return prediction_dataframe

def xgb_sequential_predict(stock_object, n_estimators, max_depth, learning_rate, min_child_weight, subsample):
  x_train_set0, y_train_set0, x_test_set0, y_test_set0, scaler = get_x_y_test_train(stock_object, -1, 0)
  test_prediction = []
  test_prediction_unscaled = []
  predict_var = stock_object.input_args.predict_var

  for i in range(len(stock_object.test_set.index)):
    #Select train and test sets and train
    x_train_set, y_train_set, x_test_set, y_test_set, scaler = get_x_y_test_train(stock_object, 1, i)
    model = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, min_child_weight = min_child_weight, subsample = subsample, colsample_bytree = stock_object.input_args.col_std, colsample_bylevel = stock_object.input_args.col_mean)
    model.fit(x_train_set.values, y_train_set.values)

    #Predict stock price
    prediction = float(model.predict(x_test_set.iloc[0].values.reshape(1,-1)))
    test_prediction.append(prediction)
    #Calculate Unscaled Prediction
    prediction_array = x_test_set.iloc[0]
    estimate_array = pd.Series([prediction], index = [predict_var])
    prediction_array = prediction_array.append(estimate_array)
    unscaled_df = scaler.inverse_transform(prediction_array)
    test_prediction_unscaled.append(unscaled_df[-1])
    
  test_prediction = np.asarray(test_prediction)
  test_prediction_unscaled = np.asarray(test_prediction_unscaled)
  print('len test_set {}'.format(len(x_test_set)))
  print('len test_prediction {}'.format(len(test_prediction)))

  #scaled_prediction_set = get_prediction_dataframe(x_test_set0, test_prediction, stock_object)
  print('test set test')
  print(x_test_set0)
  print(len(x_test_set0))
  print('len prediction')
  print(len(test_prediction_unscaled))
  unscaled_prediction_set = get_prediction_dataframe(x_test_set0, test_prediction_unscaled, stock_object)

  #scaled_weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set0.to_numpy().tolist())
  scaled_weekly_total_error = 0
  unscaled_weekly_total_error = get_general_errors_dataframes(test_prediction_unscaled, y_test_set0.to_numpy().tolist())
  print('UnScaled XGB error: {}'.format(unscaled_weekly_total_error))
  #print('Scaled XGB error: {}'.format(scaled_weekly_total_error))

  #prediction_overlay_plot(stock_object.test_set, stock_object.train_set, scaled_prediction_set, scaled_weekly_total_error, 'SeqXGBScaled', stock_object)
  prediction_overlay_plot(stock_object.test_set_unscaled, stock_object.train_set_unscaled, unscaled_prediction_set, unscaled_weekly_total_error, 'SeqXGBUnScaled', stock_object)

  stock_object.add_unscaled_model("SeqXGBUnScaled", unscaled_prediction_set, unscaled_weekly_total_error)
  #stock_object.add_scaled_model("SeqXGBScaled", scaled_prediction_set, scaled_weekly_total_error)

  return scaled_weekly_total_error, unscaled_weekly_total_error
  
def xgb_predict(stock_object, n_estimators, max_depth, learning_rate, min_child_weight, subsample):
  x_train_set, y_train_set, x_test_set, y_test_set, scaler = get_x_y_test_train(stock_object, 0, 0)
  model = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, min_child_weight = min_child_weight, subsample = subsample, colsample_bytree = stock_object.input_args.col_std, colsample_bylevel = stock_object.input_args.col_mean)
  model.fit(x_train_set, y_train_set)

  test_prediction = model.predict(x_test_set)
  scaled_prediction_set = get_prediction_dataframe(x_test_set, test_prediction, stock_object)
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

  myfit = np.polyfit(stock_object.train_set[stock_object.input_args.date_name], stock_object.train_set[stock_object.input_args.predict_var], n)

  x_train_set, y_train_set, x_test_set, y_test_set, scaler = get_x_y_test_train(stock_object, 0)
  test_prediction = model(x_test_set[stock_object.input_args.date_name])
  scaled_prediction_set = get_prediction_dataframe(x_test_set, test_prediction, stock_object)
  unscaled_prediction_set = get_unscaled_data(scaled_prediction_set, stock_object)

  scaled_weekly_total_error = get_general_errors_dataframes(scaled_prediction_set[stock_object.input_args.predict_var], stock_object.test_set[stock_object.input_args.predict_var])
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
    plt.plot(model[stock_object.input_args.date_name], model[stock_object.input_args.predict_var], label = '{} Fit RMSE = {}'.format(name, rmse))
  plt.plot(stock_object.test_set_unscaled[stock_object.input_args.date_name], stock_object.test_set_unscaled[stock_object.input_args.predict_var], label = 'Test set')

  plt.legend()
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': All UnScaled Models')
  plt.legend(loc='upper left', borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/stock_models_all_unscaled_' + stock_object.stock_name + '_overlay.pdf')    
  plt.close('all')

  #Overlay Scaled Predictions, Test, and Train Sets
  for name, model, rmse in zip(stock_object.scaled_model_names, stock_object.scaled_models, stock_object.scaled_errors):
    plt.plot(model[stock_object.input_args.date_name], model[stock_object.input_args.predict_var], label = '{} Fit RMSE = {}'.format(name, rmse))
  plt.plot(stock_object.test_set[stock_object.input_args.date_name], stock_object.test_set[stock_object.input_args.predict_var], label = 'Test set')

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



