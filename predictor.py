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
model_lstmonetoone = load_model("modelos/vanilla_lstm_mdl.h5")
model_lstmonetomany = load_model("modelos/stkdlstm_one2many_simplevar_mdl.h5")


def randomforest(features):
	pred = clf_rf.predict(features)
	return pred

def decisiontrees(features):
	pred = clf_dt.predict(features)
	return pred


def lstm_mid_onetoone(features):
	values = features
	n_samples = values.shape[0]
	n_steps = 1
	n_features = values.shape[1]
	values = values.astype('float32')
	scaled = scaler_std.transform(values)
	test_X = scaled.reshape((n_samples, n_steps, n_features))
	yhat = model_lstmonetoone.predict(test_X)
	inv_yhat = scaler_std.inverse_transform(yhat)
	return inv_yhat


def lstm_mid_onetomany(features):


	return inv_yhat
