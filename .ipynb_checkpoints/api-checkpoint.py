from flask import Flask,request,jsonify
from sklearn.model_selection import train_test_split
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
import numpy as np
keras = tf.keras


app = Flask(__name__)
CORS(app)

existing_data = pd.read_csv('data/diabetes.csv')

x = existing_data.drop(columns=['Pregnancies', 'BloodPressure', 'SkinThickness', 'Outcome'], axis=1)
y = existing_data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=2019)
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=2019)

means = np.mean(x_train, axis=0)
stds = np.std(x_train, axis=0)

model = keras.models.load_model('./model/diabetes_model.h5')

@app.route("/predict",methods=['POST'])
def predict():
    data = request.json["data"]
    print(data)
    params = data.split(',')
    print(len(params))
    print(params[0])

    patient = {'glucose': [float(params[0])], 
                'insuline': [float(params[1])],
                'bmi': [float(params[2])], 
                'history': [float(params[3])], 
                'age': [float(params[4])],
                }
    
#     patient = [{'glucose': 0.8999,
#                 'insuline': -0.691,
#                 'bmi': 2.21654, 
#                 'history': -0.345, 
#                 'age': 0.526}]
    df = pd.DataFrame.from_dict(patient)
    print("Dataframe created: {}".format(df))
    df = (df - means[0]) / stds[0]
    print("Normalized: {}".format(df))
    print("Data: {}".format(df[0:1]))
    print("For predict {}".format(df.iloc[0]))
#     return("response")
    try:
        print("Before predict")
        out = model.predict(df[0:1])
        print("Prediction: {}".format(out))
        result = np.argmax(out)
        print("Result: {}".format(result))
        return jsonify({"result":str(200)})
    
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000, debug=True)