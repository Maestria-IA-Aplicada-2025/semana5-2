from flask import Flask, request, jsonify
import joblib
import pandas as pd
import string
import scipy
import pickle
import nltk
nltk.download('punkt_tab')

app = Flask(__name__)

@app.route('/predict/<mensaje>', methods=['POST', 'GET'])
def predict(mensaje=None):
    """
    """
    # mensaje = request.args.get('mensaje')
    print(mensaje)
    m = vector.transform([mensaje])
    prediction = clf.predict(m)
    return jsonify({'prediction': list(prediction)})
     # return jsonify({'prediction': mensaje})

if __name__ == '__main__':
    clf = joblib.load('model01.pkl')
    # vector = joblib.load('real_vectorizer.pkl', "r")
    vector = pickle.load(open("vector.pickel", "rb"))
    app.run(port=8080)
