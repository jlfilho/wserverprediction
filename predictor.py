import numpy as np
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


scaler_StdScaler = StandardScaler()
scaler_MinMax = MinMaxScaler(feature_range=(-1, 1))

scaler_MinMax = joblib.load('modelos/MinMaxScaler_model.pkl')
scaler_StdScaler = joblib.load('modelos/StandardScaler_model.pkl')
model_gruthreetofive = load_model("modelos/vanillagru_three2five_maxbw-exp_mdl.h5")
model_gruthreetoone = load_model("modelos/vanillagru_three2One_maxbw-exp_mdl.h5")

def gru_mid_threetofive(features):
	values = features
	values = values.astype('float32')
	scaled = scaler_MinMax.transform(values)
	X = np.array(scaled)
	# make forecasts
	y_hat = forecast_threetofive(model_gruthreetofive, X, n_batch=1)
	# inverse transform forecasts and test
	y_hat = inverse_transform(y_hat, scaler_MinMax)
	return y_hat

def gru_mid_threetoone(features):
	values = features
	values = values.astype('float32')
	scaled = scaler_StdScaler.transform(values)
	X = np.array(scaled)
	# make forecasts
	y_hat = forecast_threetoone(model_gruthreetoone, X, n_batch=1)
	# inverse transform forecasts and test
	y_hat = inverse_transform(y_hat, scaler_StdScaler)
	return y_hat

# make one forecast with an GRU threetoone,
def forecast_threetoone(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, len(X),1)
	# make forecast
	forecast = model.predict(X)
	# convert to array
	return [x for x in forecast[0, :]]

# make one forecast with an GRU threetofive,
def forecast_threetofive(model, X, n_batch):
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
