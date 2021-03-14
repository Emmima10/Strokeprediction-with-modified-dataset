import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler as scaler
from tensorflow import keras



app = Flask(__name__)
#scaler = pickle.load(open('scaler.pkl', 'rb'))
#model =(open('new_model.h5','rb'))
model = keras.models.load_model("new_model.h5")
scaler = joblib.load('sc_scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    #final_features =final_features.reshape(1,-1)
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = (prediction[0])
    print(output)
    if (output)==1:
        predicition_text="ABNORMAL"
    else:
        prediction_text="NORMAL"

   # if output == 0:
        return render_template('index.html', 'prediction_text')
    
    #else:
        # return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A STROKE')
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)