#!/usr/bin/env python
import re, json
from bottle import request, response, route
from predictor import gru_mid_threetoone
from predictor import gru_mid_threetofive
import numpy as np
from sklearn.externals import joblib

@route ('/downthrpt', method='POST')
def downthrpt():
     response.content_type = 'application/json'
     features = request.json['features']
     model = request.json['model']
     
     if(model == 'gru_mid_threetofive'):
        prediction = gru_mid_threetofive(features = np.array(features).reshape(3,1))
        prediction = prediction[0]
     elif(model == 'gru_mid_threetoone'):
        prediction = gru_mid_threetoone(features = np.array(features).reshape(3,1))
        prediction = prediction[0]
     else:
        prediction = np.empty(shape=[0, 1])
     return json.dumps(prediction.tolist())
