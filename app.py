from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import keras
from keras.models import load_model
global graph,model
import tensorflow as tf

# Flask
app = Flask(__name__)

# Load model
#graph = tf.get_default_graph()
#model = load_model('NewPlayerNeuralNet.h5')
#sc = joblib.load('scaler.joblib')

#@app.route("/",methods=['GET'])
#def hello():
#    return "Hello Worldsss!!!=="

@app.route("/",methods=['GET','POST'])
def hello():
    if request.method=='GET':
        return('Welcome')
    else:
        return ('Posting?')


#if __name__ == '__server__':
#    app.run(host="localhost", port=5000)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
