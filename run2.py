
from flask import Flask, request, jsonify  # Flask es un framework para crear aplicaciones web; request maneja solicitudes HTTP; jsonify convierte respuestas a formato JSON
import joblib  # joblib se usa para cargar el modelo de machine learning previamente entrenado y guardado
import pandas as pd  # pandas se usa para manejar y analizar datos en forma de DataFrames (aunque no se utiliza directamente en este código)
import string  # string proporciona funciones útiles para trabajar con cadenas de caracteres (como eliminar puntuación, etc.)
import scipy  # scipy es útil para operaciones científicas y matemáticas (aunque no se usa directamente en este código)
import pickle  # pickle es utilizado para cargar objetos serializados, en este caso el vectorizador
import nltk  # nltk es una librería para procesamiento de lenguaje natural (NLP)
nltk.download('punkt')  # Descarga el conjunto de datos 'punkt' de NLTK, necesario para tokenización de texto

# Crear una instancia de la aplicación Flask
app = Flask(__name__)  # Flask se utiliza para crear la aplicación web, __name__ es el nombre del módulo actual

# Ruta de la predicción, donde se recibe un mensaje a través de la URL
@app.route('/predict/<mensaje>', methods=['POST', 'GET'])  # La ruta '/predict/<mensaje>' recibe el mensaje en la URL, con métodos POST y GET
def predict(mensaje=None):  # La función 'predict' recibe el mensaje de la URL como argumento
    # Esta función recibe un mensaje y devuelve la predicción realizada por el modelo de machine learning.
    
    # Se imprime el mensaje recibido para depuración
    print(mensaje)
    
    # El vectorizador transforma el mensaje recibido en el formato adecuado para el modelo
    m = vector.transform([mensaje])  # 'mensaje' se transforma en una representación numérica adecuada para el modelo
    
    # El modelo realiza la predicción sobre el mensaje transformado
    prediction = clf.predict(m)  # Se realiza la predicción utilizando el modelo cargado
    
    # Devuelve la predicción en formato JSON
    return jsonify({'prediction': list(prediction)})  # La predicción se convierte en una lista y se devuelve como respuesta JSON
    
    # Opción comentada para devolver el mensaje tal cual, en lugar de la predicción
    # return jsonify({'prediction': mensaje})

# Esta parte asegura que el código solo se ejecute cuando se ejecuta directamente (no cuando se importa como módulo)
if __name__ == '__main__':
    # Carga del modelo de machine learning desde un archivo pickle
    clf = joblib.load('model01.pkl')  # Se carga el modelo previamente entrenado usando joblib
    
    # Carga del vectorizador (que convierte el texto en vectores numéricos) desde un archivo pickle
    vector = pickle.load(open("vector.pickel", "rb"))  # Se carga el vectorizador previamente guardado con pickle
    
    # Se ejecuta la aplicación Flask en el puerto 8080
    app.run(port=8080)  # Flask inicia la aplicación en el puerto 8080 para que esté disponible para recibir solicitudes
