from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle as p

app = Flask(__name__)

modelfile = 'models/diabetes_final.pickle'    

model = p.load(open(modelfile, 'rb'))

@app.route('/')
def main():
    return ('Predict diabetes API')
    
@app.route('/api/', methods=['POST'])
def makecalc():
	j_data = request.get_json()

	prediction = np.array2string(model.predict(j_data))
	
	return jsonify(prediction)
