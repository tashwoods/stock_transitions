from imported_libraries import *

def get_x_y_test_train(stock_object, scale, i):
  predict_var = stock_object.input_args.predict_var
  date_name = stock_object.input_args.date_name
  scaler = StandardScaler()

  if scale == -1: #do no scale at all
    train_set = stock_object.train_set_unscaled
    test_set = stock_object.test_set_unscaled

  else:
    if scale == 0: #scale test set to train set
      train_set = stock_object.train_set_unscaled
      test_set = stock_object.test_set_unscaled

    if scale == 1: #iteratively scale test set to train set
      train_set = stock_object.train_set_unscaled
      test_set = stock_object.test_set_unscaled
      if i != 0:
        if i == 1:
          append = test_set.iloc[0]
        else:
          append = test_set.iloc[:i]
        train_set = train_set.append(append)
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

  return test_set, train_set, x_train_set, y_train_set, x_test_set, y_test_set, scaler

def get_unscaled_data(dataset, stock_object, scaler):
  array_data = scaler.inverse_transform(dataset)
  formatted_data = pd.DataFrame(array_data, index = dataset.index, columns = dataset.columns)
  return formatted_data

def get_prediction_dataframe(x_test_set, test_prediction, stock_object):
  test_prediction_scaled = pd.DataFrame(test_prediction)
  prediction_dataframe = x_test_set
  prediction_dataframe[stock_object.input_args.predict_var] = test_prediction_scaled.values
  return prediction_dataframe

def xgb_sequential_predict(stock_object, n_estimators, max_depth, learning_rate, min_child_weight, subsample):
  test_set0, train_set0, x_train_set0, y_train_set0, x_test_set0, y_test_set0, scaler = get_x_y_test_train(stock_object, -1, 0)
  predict_var = stock_object.input_args.predict_var
  test_prediction = []
  test_prediction_unscaled = []
  #iterate through test set
  for i in range(len(stock_object.test_set_unscaled.index)):
    #Iteratively augment, scale and model test set
    test_set, train_set, x_train_set, y_train_set, x_test_set, y_test_set, scaler = get_x_y_test_train(stock_object, 1, i)
    model = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, min_child_weight = min_child_weight, subsample = subsample)
    model.fit(x_train_set.values, y_train_set.values)
    #Predict stock price for only first entry in test set, as this is an iterative BDT
    prediction_array = x_test_set.iloc[0]
    prediction = float(model.predict(prediction_array.values.reshape(1,-1)))
    test_prediction.append(prediction)
    #Calculate Unscaled Prediction
    estimate_array = pd.Series([prediction], index = [predict_var])
    prediction_array = prediction_array.append(estimate_array)
    unscaled_df = scaler.inverse_transform(prediction_array)
    test_prediction_unscaled.append(unscaled_df[-1]) #only add predict var estimate to array

    #print('feature importance')
    #feature_dict = {}
    #for col, score in zip(stock_object.train_set_unscaled.columns, model.feature_importances_):
    #  feature_dict[col] = score
    #  print(col,score)
    #ordered_feature_list = sorted(feature_dict, key = feature_dict.get, reverse=True)
    #for 
    #print(ordered_feature_list)
    
  test_prediction = np.asarray(test_prediction)
  test_prediction_unscaled = np.asarray(test_prediction_unscaled)
  unscaled_prediction_set = get_prediction_dataframe(x_test_set0, test_prediction_unscaled, stock_object)


  #Obtain and Save Modelling Errors
  unscaled_weekly_total_error = get_general_errors_dataframes(test_prediction_unscaled, y_test_set0.to_numpy().tolist())
  print('UnScaled XGB error: {}'.format(unscaled_weekly_total_error))
  name = "SeqXGBUnScaled_E"+ str(n_estimators) + "_E" + str(max_depth) + "_L" + str(learning_rate) + "_M" + str(min_child_weight) + "_S" + str(subsample)
  stock_object.add_unscaled_model(name, unscaled_prediction_set, unscaled_weekly_total_error)
  #Plot
  prediction_overlay_plot(stock_object.test_set_unscaled, stock_object.train_set_unscaled, unscaled_prediction_set, unscaled_weekly_total_error, 'SeqXGBUnScaled', stock_object)

  return unscaled_weekly_total_error
  
def xgb_predict(stock_object, n_estimators, max_depth, learning_rate, min_child_weight=1, subsample=1):
  #Scale, split and model data
  predict_var = stock_object.input_args.predict_var
  test_set, train_set, x_train_set, y_train_set, x_test_set, y_test_set, scaler = get_x_y_test_train(stock_object, 0, 0)
  model = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, min_child_weight = min_child_weight, subsample = subsample)
  model.fit(x_train_set.values, y_train_set.values)
  test_prediction = model.predict(x_test_set.values)
  scaled_prediction_set = get_prediction_dataframe(x_test_set, test_prediction, stock_object)
  unscaled_prediction_set = get_unscaled_data(scaled_prediction_set, stock_object, scaler)

  #Print and Save Modelling Errors
  scaled_weekly_total_error = get_general_errors_dataframes(test_prediction, y_test_set.to_numpy().tolist())
  unscaled_weekly_total_error = get_general_errors_dataframes(unscaled_prediction_set[predict_var], stock_object.test_set_unscaled[predict_var])
  stock_object.add_unscaled_model("XGBUnScaled", unscaled_prediction_set, unscaled_weekly_total_error)
  print('UnScaled XGB error: {}'.format(unscaled_weekly_total_error))
  print('Scaled XGB error: {}'.format(scaled_weekly_total_error))

  #Plot Results 
  prediction_overlay_plot(stock_object.test_set_unscaled, stock_object.train_set_unscaled, unscaled_prediction_set, unscaled_weekly_total_error, 'XGBUnScaled', stock_object)
  prediction_overlay_plot(test_set, train_set, scaled_prediction_set, scaled_weekly_total_error, 'XGBScaled', stock_object)

  print('feature importance')
  for col, score in zip(stock_object.train_set_unscaled.columns, model.feature_importances_):
    print(col,score)

  return mean_squared_error(y_test_set, test_prediction)

def poly_fit(stock_object, n):
  #Simplify variable names
  predict_var = stock_object.input_args.predict_var
  date_name = stock_object.input_args.date_name

  #Scale, split, model data
  test_set, train_set, x_train_set, y_train_set, x_test_set, y_test_set, scaler = get_x_y_test_train(stock_object, 0, 0)
  model = np.poly1d(np.polyfit(train_set[date_name], train_set[predict_var], n))
  test_prediction = model(x_test_set[date_name])
  scaled_prediction_set = get_prediction_dataframe(x_test_set, test_prediction, stock_object)
  unscaled_prediction_set = get_unscaled_data(scaled_prediction_set, stock_object, scaler)

  #Obtain and Save Modelling Errors
  scaled_weekly_total_error = get_general_errors_dataframes(scaled_prediction_set[predict_var], test_set[predict_var])
  unscaled_weekly_total_error = get_general_errors_dataframes(unscaled_prediction_set[predict_var], stock_object.test_set_unscaled[predict_var])
  stock_object.add_unscaled_model('Poly' + str(n) + 'UnScaled' , unscaled_prediction_set, unscaled_weekly_total_error)
  #stock_object.add_scaled_model('Poly' + str(n) + 'Scaled', scaled_prediction_set, scaled_weekly_total_error)
  print('SCALED: Degree: {} Weekly error: {}'.format(n, scaled_weekly_total_error))
  print('UNSCALED: Degree: {} Weekly error: {}'.format(n, unscaled_weekly_total_error))

  #Plot Results
  prediction_overlay_plot(stock_object.test_set_unscaled, stock_object.train_set_unscaled, unscaled_prediction_set, unscaled_weekly_total_error, str(n) + 'PolyUnscaled', stock_object) 
  prediction_overlay_plot(test_set, train_set, scaled_prediction_set, scaled_weekly_total_error, str(n) + 'PolyScaled', stock_object)

def overlay_predictions(stock_object):
  #Overlay Unscaled Predictions, Test, and Train Sets
  alpha = 0.8
  linewidth = 0.8
  leg_loc = 'lower right'
  for name, model, rmse in zip(stock_object.unscaled_model_names, stock_object.unscaled_models, stock_object.unscaled_errors):
    plt.plot(model[stock_object.input_args.date_name], model[stock_object.input_args.predict_var], label = '{} Fit RMSE = {}'.format(name, rmse), alpha = alpha, linewidth = linewidth)
  plt.plot(stock_object.test_set_unscaled[stock_object.input_args.date_name], stock_object.test_set_unscaled[stock_object.input_args.predict_var], label = 'Test set')

  plt.legend()
  gca().get_xaxis().get_major_formatter().set_useOffset(False)
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': All UnScaled Models')
  plt.legend(loc= leg_loc, borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/stock_models_all_unscaled_' + stock_object.stock_name + '_overlay.pdf')    
  plt.close('all')

  '''
  #Overlay Scaled Predictions, Test, and Train Sets
  for name, model, rmse in zip(stock_object.scaled_model_names, stock_object.scaled_models, stock_object.scaled_errors):
    plt.plot(model[stock_object.input_args.date_name], model[stock_object.input_args.predict_var], label = '{} Fit RMSE = {}'.format(name, rmse), alpha = alpha, linewidth = linewidth)
  plt.plot(stock_object.test_set[stock_object.input_args.date_name], stock_object.test_set[stock_object.input_args.predict_var], label = 'Test set')

  plt.legend()
  plt.xlabel(stock_object.input_args.date_name)
  plt.ylabel(stock_object.input_args.predict_var)
  plt.title(stock_object.input_args.predict_var + ' vs. ' + stock_object.input_args.date_name + ': All Scaled Models')
  plt.legend(loc = leg_loc, borderaxespad=0., prop={'size': 6})
  plt.savefig(stock_object.input_args.output_dir + '/' + stock_object.stock_name + '/stock_models_all_scaled_' + stock_object.stock_name + '_overlay.pdf')    
  plt.close('all')
  '''

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



