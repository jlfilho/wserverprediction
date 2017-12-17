import numpy as np
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

scaler_std = StandardScaler()
scaler_std = joblib.load('modelos/StandardScaler_model.pkl')
scaler_MinMax = MinMaxScaler(feature_range=(-1, 1))
scaler_MinMax = joblib.load('modelos/MinMaxScaler_model.pkl')
model_lstmonetoone = load_model("modelos/vanillalstm_One2One_abw9-3_mdl.h5")
model_lstmonetomany = load_model("modelos/stkdlstm_one2many_abw9-3_mdl.h5")


def randomforest(features):
	pred = clf_rf.predict(features)
	return pred

def decisiontrees(features):
	pred = clf_dt.predict(features)
	return pred


def lstm_mid_threetoone(features):
	values = features
	n_samples = values.shape[0]
	n_steps = 3
	n_features = values.shape[1]
	values = values.astype('float32')
	scaled = scaler_std.transform(values)
	test_X = scaled.reshape((-1, n_steps, n_features))
	yhat = model_lstmonetoone.predict(test_X)
	inv_yhat = scaler_std.inverse_transform(yhat)
	return inv_yhat


def lstm_mid_onetomany(features):
	values = features
	values = values.astype('float32')
	scaled = scaler_MinMax.transform(values)
	X = np.array(scaled)
	# make forecasts
	y_hat = forecast_lstm(model_lstmonetomany, X, n_batch=1)
	# inverse transform forecasts and test
	y_hat = inverse_transform(y_hat, scaler_MinMax)
	return y_hat


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# inverse data transform on forecasts
def inverse_transform(forecasts, scaler):
	inverted = list()
    # create array from forecast
	forecast = np.array(forecasts)
	forecast = forecast.reshape(1, len(forecast))
	# invert scaling
	inv_scale = scaler.inverse_transform(forecast)
	inv_scale = inv_scale[0, :]
	# store
	inverted.append(inv_scale)
	return inverted
